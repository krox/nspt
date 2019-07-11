#ifndef UTIL_HDF5_H
#define UTIL_HDF5_H

/**
 * Simple C++ wrapper for HDF5.
 */

#include <hdf5.h>
#include <string>

#include "util/span.h"
#include "util/vector2d.h"

namespace util {

class DataObject
{
  private:
	void setAttribute(const std::string &, hid_t, const void *);
	void setAttribute(const std::string &, hid_t, hsize_t, const void *);

	void getAttribute(const std::string &name, hid_t type, void *data);

  protected:
	hid_t id = 0;

	DataObject() = default;
	DataObject(hid_t id) : id(id) {}

  public:
	/** attributes */
	void setAttribute(const std::string &name, double v);
	void setAttribute(const std::string &name, int v);
	void setAttribute(const std::string &name, const std::string &v);
	void setAttribute(const std::string &name, span<const double> v);
	void setAttribute(const std::string &name, span<const int> v);

	template <typename T> T getAttribute(const std::string &name);
};

class DataSet : public DataObject
{
	size_t size_ = 0;
	std::vector<hsize_t> shape_;

  public:
	size_t size() const { return size_; }
	const std::vector<hsize_t> &shape() const { return shape_; }
	size_t rank() const { return shape_.size(); }

	/** non copyable but movable */
	DataSet(const DataSet &) = delete;
	DataSet &operator=(const DataSet &) = delete;
	DataSet(DataSet &&f) : DataObject(f.id), size_(f.size_), shape_(f.shape_)
	{
		f.id = 0;
	};
	DataSet &operator=(DataSet &&f)
	{
		close();
		id = f.id;
		size_ = f.size_;
		shape_ = f.shape_;
		f.id = 0;
		return *this;
	}

	DataSet() = default;
	explicit DataSet(hid_t id);
	~DataSet();
	void close();

	/** write the whole dataset */
	void write(span<const double> data);

	/** write a single "row" (i.e. first index) */
	void write(hsize_t row, span<const double> data);

	/** read the whole dataset */
	void read(span<double> data);
	template <typename T> std::vector<T> read();
};

class DataFile : public DataObject
{
	explicit DataFile(hid_t id_) : DataObject(id_) {}

  public:
	/** non copyable but movable */
	DataFile(const DataFile &) = delete;
	DataFile &operator=(const DataFile &) = delete;
	DataFile(DataFile &&f) : DataObject(f.id) { f.id = 0; };
	DataFile &operator=(DataFile &&f)
	{
		close();
		id = f.id;
		f.id = 0;
		return *this;
	}

	/** open/close */
	DataFile() = default;
	~DataFile();
	static DataFile create(const std::string &filename, bool overwrite = false);
	static DataFile open(const std::string &filename, bool writeable = false);
	void close();

	/** general object access */
	bool exists(const std::string &name);
	void remove(const std::string &name);

	/** access to datasets */
	DataSet createData(const std::string &name,
	                   const std::vector<hsize_t> &size);
	DataSet openData(const std::string &name);

	/** convenience methods combining createData + write */
	DataSet createData(const std::string &name,
	                   const std::vector<hsize_t> &size,
	                   span<const double> data);
	DataSet createData(const std::string &name, span<const double> data);
	DataSet createData(const std::string &name, const vector2d<double> &data);

	/** groups */
	void makeGroup(const std::string &name);
};

} // namespace util

#endif
