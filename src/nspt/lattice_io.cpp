#include "nspt/lattice_io.h"

#include "nlohmann/json.hpp"

using namespace nlohmann;
using namespace Grid;

#define MAGIC 3113769617510792126

struct Header
{
	uint64_t magic = MAGIC;
	uint64_t flags = 0;
	uint32_t meta_size = 0;
	uint32_t meta_crc = 0;
	uint64_t data_size = 0;
	uint64_t unused_[4];
};

static_assert(sizeof(Header) == 64);

template <typename vobj>
static void computeChecksum(const Lattice<vobj> &U, uint32_t &scidac_csuma,
                            uint32_t &scidac_csumb)
{
	auto grid = U._grid;
	auto data = std::vector<typename vobj::scalar_object>(grid->lSites());
	unvectorizeToLexOrdArray(data, U);
	scidac_csuma = 0;
	scidac_csumb = 0;
	BinaryIO::ScidacChecksum(grid, data, scidac_csuma, scidac_csumb);
}

/** grid size and checksums will be added to the json */
template <typename vobj>
static void writeField(const std::string &filename, Lattice<vobj> &U,
                       json &meta)
{
	using sobj = typename vobj::scalar_object;
	uint32_t scidac_csuma, scidac_csumb;
	computeChecksum(U, scidac_csuma, scidac_csumb);

	// prepare header and meta data
	meta["grid"] = U._grid->FullDimensions();
	meta["scidac_csuma"] = scidac_csuma;
	meta["scidac_csumb"] = scidac_csumb;
	std::string meta_str = meta.dump();
	meta_str.resize((meta_str.size() + 63ULL) & ~63ULL, char(0));
	Header header;
	header.meta_size = meta_str.size();
	header.meta_crc = GridChecksum::crc32(meta_str.data(), meta_str.size());
	header.data_size = U._grid->gSites() * sizeof(sobj);

	// write header and meta-data
	if (primaryTask())
	{
		fmt::print("writing {}\n", filename);
		fmt::print("grid = {}\n",
		           util::span<const int>(U._grid->FullDimensions()));
		fmt::print("meta-crc32 = {}, data-scidac-csum = {} {}\n",
		           header.meta_crc, scidac_csuma, scidac_csumb);

		auto file = std::fstream(filename, std::ios::out | std::ios::binary);
		file.write((const char *)&header, sizeof(Header));
		file.write(meta_str.data(), meta_str.size());
		file.close();
	}

	// write data
	size_t offset = sizeof(Header) + meta_str.size();
	assert(offset % 64 == 0);

	std::string format = "IEEE64"; // little-endian double-precision
	auto munge = [](const sobj &x, sobj &y) { y = x; }; // no conversion
	uint32_t _1, _2, _3; // unused checksum output
	BinaryIO::writeLatticeObject<vobj, sobj>(U, filename, munge, offset, format,
	                                         _1, _2, _3);
}

void writeConfig(const std::string &filename, QCD::LatticeGaugeField &U)
{
	json meta;
	meta["type"] = "LatticeLorentzColourMatrix";
	writeField(filename, U, meta);
}

std::vector<int> getGridFromFile(const std::string &filename)
{
	Header header;
	std::string meta_str;

	// I hope openeing from all tasks in parallel works
	{
		auto file = std::fstream(filename, std::ios::in | std::ios::binary);
		file.read((char *)&header, sizeof(Header));
		meta_str.resize(header.meta_size);
		file.read(meta_str.data(), meta_str.size());
		file.close();
	}

	if (header.magic != MAGIC)
		throw std::runtime_error("magic number mismatch");

	json meta = json::parse(meta_str);
	std::vector<int> file_grid;
	meta.at("grid").get_to(file_grid);
	return file_grid;
}

/** grid size and checksums will be added to the json */
template <typename vobj>
static void readField(const std::string &filename, Lattice<vobj> &U, json &meta)
{
	Header header;
	std::string meta_str;
	using sobj = typename vobj::scalar_object;

	// I hope openeing from all tasks in parallel works
	{
		auto file = std::fstream(filename, std::ios::in | std::ios::binary);
		file.read((char *)&header, sizeof(Header));
		meta_str.resize(header.meta_size);
		file.read(meta_str.data(), meta_str.size());
		file.close();
	}
	if (header.magic != MAGIC)
		throw std::runtime_error("magic number mismatch");

	uint32_t file_scidac_csuma, file_scidac_csumb;
	std::vector<int> file_grid;
	meta = json::parse(meta_str);
	meta.at("grid").get_to(file_grid);
	meta.at("scidac_csuma").get_to(file_scidac_csuma);
	meta.at("scidac_csumb").get_to(file_scidac_csumb);

	if (primaryTask())
	{
		fmt::print("reading {}\n", filename);
		fmt::print("grid = {}\n", util::span<const int>(file_grid));
		fmt::print("meta-crc32 = {}, data-scidac-csum = {} {}\n",
		           header.meta_crc, file_scidac_csuma, file_scidac_csumb);
	}
	if (header.meta_crc !=
	    GridChecksum::crc32(meta_str.data(), meta_str.size()))
		throw std::runtime_error("meta-crc32 mismatch");
	if (file_grid != U._grid->FullDimensions())
		throw std::runtime_error("Lattice size mismatch");

	size_t offset = sizeof(Header) + meta_str.size();
	assert(offset % 64 == 0);
	uint32_t scidac_csuma = 0, scidac_csumb = 0, _ = 0;
	std::string format = "IEEE64"; // little-endian double-precision
	auto munge = [](const sobj &x, sobj &y) { y = x; }; // no conversion
	BinaryIO::readLatticeObject<vobj, sobj>(U, filename, munge, offset, format,
	                                        _, scidac_csuma, scidac_csumb);

	if (file_scidac_csuma != scidac_csuma || file_scidac_csumb != scidac_csumb)
		throw std::runtime_error("data checksum mismatch");
}

void readConfig(const std::string &filename, QCD::LatticeGaugeField &U)
{
	json meta;
	readField(filename, U, meta);
	assert(meta["type"] == "LatticeLorentzColourMatrix");
}
