
#include "util/hdf5.h"

#include <cassert>
#include <stdexcept>

namespace util {

namespace {
hid_t enforce(hid_t id)
{
	if (id < 0)
		throw std::runtime_error("HDF5 error");
	return id;
}
} // namespace

DataObject::~DataObject() {}

DataSet::DataSet(hid_t id) : DataObject(id)
{
	if (id_ == 0)
		return;
	auto space = enforce(H5Dget_space(id_));
	size_ = H5Sget_simple_extent_npoints(space);
	shape_.resize(64);
	auto rank = H5Sget_simple_extent_dims(space, &shape_[0], nullptr);
	shape_.resize(rank);
	enforce(H5Sclose(space));
}

void DataSet::close()
{
	if (id_ == 0)
		return;
	enforce(H5Dclose(id_));
	id_ = 0;
}

DataSet::~DataSet() { close(); }

void DataSet::read(span<double> data)
{
	assert(data.size() == size_);
	enforce(H5Dread(id_, H5T_NATIVE_DOUBLE, 0, 0, 0, data.data()));
}

template <> std::vector<double> DataSet::read<double>()
{
	auto r = std::vector<double>(size_);
	read(r);
	return r;
}

void DataSet::write(span<const double> data)
{
	assert(data.size() == size_);
	enforce(H5Dwrite(id_, H5T_NATIVE_DOUBLE, 0, 0, 0, data.data()));
}

void DataSet::write(hsize_t row, span<const double> data)
{
	assert(id_ > 0);
	assert(row < shape_[0]);
	assert(data.size() == size_ / shape_[0]);

	auto offset = std::vector<hsize_t>(shape_.size(), 0);
	offset[0] = row;
	auto memspace =
	    enforce(H5Screate_simple(shape_.size() - 1, &shape_[1], nullptr));
	std::vector<hsize_t> rowShape = shape_;
	rowShape[0] = 1;
	auto space = enforce(H5Dget_space(id_));
	H5Sselect_hyperslab(space, H5S_SELECT_SET, offset.data(), nullptr,
	                    rowShape.data(), nullptr);

	enforce(H5Dwrite(id_, H5T_NATIVE_DOUBLE, memspace, space, 0, data.data()));

	enforce(H5Sclose(memspace));
	enforce(H5Sclose(space));
}

DataFile::~DataFile() { close(); }

DataFile DataFile::create(const std::string &filename, bool overwrite)
{
	auto mode = overwrite ? H5F_ACC_TRUNC : H5F_ACC_EXCL;
	auto id = enforce(H5Fcreate(filename.c_str(), mode, 0, 0));
	return DataFile(id);
}

DataFile DataFile::open(const std::string &filename, bool writeable)
{
	auto mode = writeable ? H5F_ACC_RDWR : H5F_ACC_RDONLY;
	auto id = enforce(H5Fopen(filename.c_str(), mode, 0));
	return DataFile(id);
}

void DataFile::close()
{
	if (id_ == 0)
		return;
	enforce(H5Fclose(id_));
	id_ = 0;
}

DataSet DataFile::createData(const std::string &name,
                             const std::vector<hsize_t> &size)
{
	assert(id_ > 0);
	auto type = H5T_NATIVE_DOUBLE;
	auto space = enforce(H5Screate_simple(size.size(), size.data(), nullptr));
	auto set = enforce(H5Dcreate2(id_, name.c_str(), type, space, 0, 0, 0));
	enforce(H5Sclose(space));
	return DataSet(set);
}

DataSet DataFile::openData(const std::string &name)
{
	assert(id_ > 0);
	auto set = enforce(H5Dopen2(id_, name.c_str(), 0));
	return DataSet(set);
}

DataSet DataFile::createData(const std::string &name,
                             const std::vector<hsize_t> &size,
                             span<const double> data)
{
	auto set = createData(name, size);
	set.write(data);
	return set;
}

DataSet DataFile::createData(const std::string &name, span<const double> data)
{
	auto set = createData(name, {data.size()});
	set.write(data);
	return set;
}

DataSet DataFile::createData(const std::string &name,
                             const vector2d<double> &data)
{
	auto set = createData(name, {data.height(), data.width()});
	set.write(data.flat());
	return set;
}

bool DataFile::exists(const std::string &name)
{
	assert(id_ > 0);
	return enforce(H5Lexists(id_, name.c_str(), 0)) > 0;
}

void DataFile::remove(const std::string &name)
{
	assert(id_ > 0);
	enforce(H5Ldelete(id_, name.c_str(), 0));
}

void DataFile::createGroup(const std::string &name)
{
	assert(id_ > 0);
	auto group = enforce(H5Gcreate2(id_, name.c_str(), 0, 0, 0));
	enforce(H5Gclose(group));
}

void DataObject::setAttribute(const std::string &name, hid_t type,
                              const void *v)
{
	assert(id_ > 0);
	auto space = enforce(H5Screate(H5S_SCALAR));
	auto attr = enforce(H5Acreate2(id_, name.c_str(), type, space, 0, 0));
	enforce(H5Awrite(attr, type, v));
	enforce(H5Aclose(attr));
	enforce(H5Sclose(space));
}

void DataObject::setAttribute(const std::string &name, hid_t type,
                              hsize_t count, const void *v)
{
	assert(id_ > 0);
	auto space = enforce(H5Screate_simple(1, &count, nullptr));
	auto attr = enforce(H5Acreate2(id_, name.c_str(), type, space, 0, 0));
	enforce(H5Awrite(attr, type, v));
	enforce(H5Aclose(attr));
	enforce(H5Sclose(space));
}

void DataObject::setAttribute(const std::string &name, double v)
{
	setAttribute(name, H5T_NATIVE_DOUBLE, &v);
}

void DataObject::setAttribute(const std::string &name, int v)
{
	setAttribute(name, H5T_NATIVE_INT, &v);
}

void DataObject::setAttribute(const std::string &name, const std::string &v)
{
	auto type = enforce(H5Tcopy(H5T_C_S1));
	enforce(H5Tset_size(type, H5T_VARIABLE));
	const char *ptr = v.c_str();
	setAttribute(name, type, &ptr);
	enforce(H5Tclose(type));
}

void DataObject::setAttribute(const std::string &name, span<const double> v)
{
	setAttribute(name, H5T_NATIVE_DOUBLE, v.size(), v.data());
}

void DataObject::setAttribute(const std::string &name, span<const int> v)
{
	setAttribute(name, H5T_NATIVE_INT, v.size(), v.data());
}

void DataObject::getAttribute(const std::string &name, hid_t type, void *data)
{
	assert(id_ > 0);
	// open attribute
	auto attr = enforce(H5Aopen(id_, name.c_str(), 0));

	// check size
	auto space = enforce(H5Aget_space(attr));
	auto size = H5Sget_simple_extent_npoints(space);
	enforce(H5Sclose(space));
	if (size != 1)
		throw std::runtime_error("HDF5 error: wrong attribute size");

	// read attribute
	enforce(H5Aread(attr, type, data));

	enforce(H5Aclose(attr));
}

template <> int DataObject::getAttribute<int>(const std::string &name)
{
	int r;
	getAttribute(name, H5T_NATIVE_INT, &r);
	return r;
}

template <> double DataObject::getAttribute<double>(const std::string &name)
{
	double r;
	getAttribute(name, H5T_NATIVE_DOUBLE, &r);
	return r;
}

template <>
std::string DataObject::getAttribute<std::string>(const std::string &name)
{
	auto type = enforce(H5Tcopy(H5T_C_S1));
	enforce(H5Tset_size(type, H5T_VARIABLE));

	char *ptr;
	getAttribute(name, type, &ptr);
	auto r = std::string(ptr);
	free(ptr);

	enforce(H5Tclose(type));
	return r;
}

template <>
std::vector<int>
DataObject::getAttribute<std::vector<int>>(const std::string &name)
{
	assert(id_ > 0);

	// open attribute
	auto attr = enforce(H5Aopen(id_, name.c_str(), 0));

	// check size
	auto space = enforce(H5Aget_space(attr));
	auto size = H5Sget_simple_extent_npoints(space);
	auto r = std::vector<int>(size);
	enforce(H5Sclose(space));

	// read attribute
	enforce(H5Aread(attr, H5T_NATIVE_INT, r.data()));

	enforce(H5Aclose(attr));
	return r;
}

template <>
std::vector<double>
DataObject::getAttribute<std::vector<double>>(const std::string &name)
{
	// open attribute
	auto attr = enforce(H5Aopen(id_, name.c_str(), 0));

	// check size
	auto space = enforce(H5Aget_space(attr));
	auto size = H5Sget_simple_extent_npoints(space);
	auto r = std::vector<double>(size);
	enforce(H5Sclose(space));

	// read attribute
	enforce(H5Aread(attr, H5T_NATIVE_DOUBLE, r.data()));

	enforce(H5Aclose(attr));
	return r;
}

} // namespace util
