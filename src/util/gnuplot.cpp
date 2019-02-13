#include "util/gnuplot.h"

#include <fmt/format.h>
#include <fstream>

namespace util {

int Gnuplot::nplotsGlobal = 0;

Gnuplot::Gnuplot(bool persist) : plotID(nplotsGlobal++)
{
	if (persist)
		pipe = popen("gnuplot -p", "w");
	else
		pipe = popen("gnuplot", "w");
	if (pipe == nullptr)
		throw "ERROR: Could not open gnuplot (is it installed?)";

	fmt::print(pipe, "set output\n");
	fmt::print(pipe, "set terminal x11\n");
	fflush(pipe);
}

Gnuplot &Gnuplot::plotFunction(const std::string &fun, const std::string &title)
{
	fmt::print(pipe, "{} {} title \"{}\"\n", nplots ? "replot" : "plot", fun,
	           (title.size() ? title : fun));
	fflush(pipe);
	++nplots;
	return *this;
}

Gnuplot &Gnuplot::plotFunction(const std::function<double(double)> &fun,
                               double a, double b, const std::string &title)
{
	auto oldStyle = style;
	style = "lines";
	std::vector<double> xs, ys;
	for (int i = 0; i <= 100; ++i)
	{
		xs.push_back(a + (b - a) * i / 100);
		ys.push_back(fun(xs.back()));
	}
	plotData(xs, ys, title);
	style = oldStyle;
	return *this;
}

Gnuplot &Gnuplot::plotData(span<const double> ys, const std::string &title)
{
	std::string filename = fmt::format("gnuplot_{}_{}.txt", plotID, nplots);
	std::ofstream file(filename);
	for (size_t i = 0; i < ys.size(); ++i)
		file << i << " " << ys[i] << "\n";
	file.flush();
	file.close();
	fmt::print(pipe, "{} '{}' using 1:2 with {} title \"{}\"\n",
	           (nplots ? "replot" : "plot"), filename, style, title);
	fflush(pipe);
	++nplots;
	return *this;
}

Gnuplot &Gnuplot::plotError(span<const double> ys, span<const double> err,
                            const std::string &title)
{
	std::string filename = fmt::format("gnuplot_{}_{}.txt", plotID, nplots);
	std::ofstream file(filename);
	for (size_t i = 0; i < ys.size(); ++i)
		file << i << " " << ys[i] << " " << err[i] << "\n";
	file.flush();
	file.close();
	fmt::print(pipe, "{} '{}' using 1:2:3 with {} title \"{}\"\n",
	           (nplots ? "replot" : "plot"), filename, "errorbars", title);
	fflush(pipe);
	++nplots;
	return *this;
}

Gnuplot &Gnuplot::plotData(span<const double> xs, span<const double> ys,
                           const std::string &title)
{
	std::string filename = fmt::format("gnuplot_{}_{}.txt", plotID, nplots);
	std::ofstream file(filename);
	assert(xs.size() == ys.size());
	for (size_t i = 0; i < xs.size(); ++i)
		file << xs[i] << " " << ys[i] << "\n";
	file.flush();
	file.close();
	fmt::print(pipe, "{} '{}' using 1:2 with {} title \"{}\"\n",
	           (nplots ? "replot" : "plot"), filename, style, title);
	fflush(pipe);
	++nplots;
	return *this;
}

Gnuplot &Gnuplot::plotError(span<const double> xs, span<const double> ys,
                            span<const double> err, const std::string &title)
{
	std::string filename = fmt::format("gnuplot_{}_{}.txt", plotID, nplots);
	std::ofstream file(filename);
	assert(xs.size() == ys.size() && xs.size() == err.size());
	for (size_t i = 0; i < ys.size(); ++i)
		file << xs[i] << " " << ys[i] << " " << err[i] << "\n";
	file.flush();
	file.close();
	fmt::print(pipe, "{} '{}' using 1:2:3 with {} title \"{}\"\n",
	           (nplots ? "replot" : "plot"), filename, "errorbars", title);
	fflush(pipe);
	++nplots;
	return *this;
}

Gnuplot &Gnuplot::plotHistogram(const Histogram &hist, const std::string &title)
{
	std::string filename = fmt::format("gnuplot_{}_{}.txt", plotID, nplots);
	std::ofstream file(filename);

	for (size_t i = 0; i < hist.bins.size(); ++i)
		file << 0.5 * (hist.mins[i] + hist.maxs[i]) << " " << hist.bins[i]
		     << "\n";
	file.flush();
	file.close();
	fmt::print(pipe, "{} '{}' using 1:2 with histeps title \"{}\"\n",
	           (nplots ? "replot" : "plot"), filename, title);
	fflush(pipe);
	nplots++;
	return *this;
}

Gnuplot &Gnuplot::hline(double y)
{
	fmt::print(pipe, "{} {} lt -1 title \"\"\n", nplots ? "replot" : "plot", y);
	fflush(pipe);
	++nplots;
	return *this;
}

Gnuplot &Gnuplot::setRangeX(double min, double max)
{
	fmt::print(pipe, "set xrange[{} : {}]\n", min, max);
	fflush(pipe);
	return *this;
}

Gnuplot &Gnuplot::setRangeY(double min, double max)
{
	fmt::print(pipe, "set yrange[{} : {}]\n", min, max);
	fflush(pipe);
	return *this;
}

Gnuplot &Gnuplot::setRangeZ(double min, double max)
{
	fmt::print(pipe, "set zrange[{} : {}]\n", min, max);
	fflush(pipe);
	return *this;
}

Gnuplot &Gnuplot::setLogScaleX()
{
	fmt::print(pipe, "set logscale x\n");
	fflush(pipe);
	return *this;
}

Gnuplot &Gnuplot::setLogScaleY()
{
	fmt::print(pipe, "set logscale y\n");
	fflush(pipe);
	return *this;
}

Gnuplot &Gnuplot::setLogScaleZ()
{
	fmt::print(pipe, "set logscale z\n");
	fflush(pipe);
	return *this;
}

Gnuplot &Gnuplot::clear()
{
	fmt::print(pipe, "clear\n");
	fflush(pipe);
	nplots = 0;
	return *this;
}

Gnuplot &Gnuplot::savefig(const std::string &filename)
{
	fmt::print(pipe, "set terminal pdf\n");
	fmt::print(pipe, "set output \"{}\"\n", filename);
	fmt::print(pipe, "replot\n");
	fflush(pipe);
	return *this;
}

} // namespace util
