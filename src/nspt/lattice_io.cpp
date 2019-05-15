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

// OpenQCD format

struct HeaderOpenQCD
{
	int Nt, Nx, Ny, Nz;
	double plaq;
};

std::vector<int> getGridFromFileOpenQCD(const std::string &filename)
{
	HeaderOpenQCD header;

	// I hope openeing from all tasks in parallel works
	{
		auto file = std::fstream(filename, std::ios::in | std::ios::binary);
		file.read((char *)&header, sizeof(HeaderOpenQCD));
		file.close();
	}

	// sanity check (should trigger on endian issues)
	assert(0 < header.Nt && header.Nt <= 1024);
	assert(0 < header.Nx && header.Nx <= 1024);
	assert(0 < header.Ny && header.Ny <= 1024);
	assert(0 < header.Nz && header.Nz <= 1024);

	return {header.Nx, header.Ny, header.Nz, header.Nt};
}

void readConfigOpenQCD(const std::string &filename, QCD::LatticeGaugeField &U)
{
	using namespace Grid::QCD;

	HeaderOpenQCD header;
	auto grid = dynamic_cast<GridCartesian *>(U._grid);
	assert(grid != nullptr);
	assert(grid->FullDimensions().size() == 4);
	int Nx = grid->FullDimensions()[0];
	int Ny = grid->FullDimensions()[1];
	int Nz = grid->FullDimensions()[2];
	int Nt = grid->FullDimensions()[3];

	std::vector<ColourMatrix> data(grid->gSites() * 4);
	// I hope openeing from all tasks in parallel works
	{
		auto file = std::fstream(filename, std::ios::in | std::ios::binary);
		file.read((char *)&header, sizeof(HeaderOpenQCD));
		if (header.Nx != Nx || header.Ny != Ny || header.Nz != Nz ||
		    header.Nt != Nt)
			throw std::runtime_error("OpenQCD file lattice size mismatch");
		file.read((char *)data.data(), data.size() * sizeof(ColourMatrix));
		file.close();
	}

	// coordinate of this process
	std::vector<int> pcoor;
	grid->ProcessorCoorFromRank(CartesianCommunicator::RankWorld(), pcoor);

	// loop over local indices
	parallel_for(int idx = 0; idx < grid->lSites(); ++idx)
	{
		// convert local index to global coordinate
		std::vector<int> lcoor, gcoor;
		grid->LocalIndexToLocalCoor(idx, lcoor);
		grid->ProcessorCoorLocalCoorToGlobalCoor(pcoor, lcoor, gcoor);

		// openQCD stores links attached to odd sites
		bool neg = (gcoor[0] + gcoor[1] + gcoor[2] + gcoor[3]) % 2 != 1;

		LorentzColourMatrix site_data;
		for (int mu = 0; mu < 4; ++mu)
		{
			// determine the site at which it is stored
			std::vector<int> c = gcoor;
			if (neg)
				c[mu] = (c[mu] + 1) % grid->FullDimensions()[mu];

			// site-index in the OpenQCD format (which usese t,x,y,z order)
			int openqcd_idx =
			    (c[3] * Nx * Ny * Nz + c[0] * Ny * Nz + c[1] * Nz + c[2]) / 2;
			int openqcd_mu = (mu + 1) % 4;

			// pick the colour-matrix out
			site_data(mu) =
			    data[8 * openqcd_idx + 2 * openqcd_mu + (neg ? 1 : 0)]();
		}

		pokeLocalSite(site_data, U, lcoor);
	}

	double plaq = ColourWilsonLoops::avgPlaquette(U);

	if (primaryTask())
	{
		fmt::print("read config {}, geom = {}x{}x{}x{}\n", filename, Nx, Ny, Nz,
		           Nt);
		fmt::print("stored plaq = {}, current plaq = {} (difference = {})\n",
		           header.plaq / 3, plaq, plaq - header.plaq / 3);
	}
}
