#include "CLI/CLI.hpp"
#include "nspt/grid_utils.h"
#include "util/gnuplot.h"
#include "util/hdf5.h"
#include "util/random.h"
#include "util/stats.h"
#include <random>

using namespace util;

class Action
{
  public:
	virtual double S(double x) const = 0;
	virtual double Sd(double x) const = 0;
	virtual double normalize(double x) const { return x; }
	virtual ~Action() = default;
};

class Harmonic : public Action
{
  public:
	virtual double S(double x) const override { return 0.5 * x * x; }
	virtual double Sd(double x) const override { return x; }
};

class DoubleWell : public Action
{
  public:
	double beta;
	explicit DoubleWell(double beta = 1.0) : beta(beta) {}
	virtual double S(double x) const override
	{
		if (x < -1.0)
			return beta * ((x + 1.0) * (x + 1.0));
		if (x > 1.0)
			return beta * ((x - 1.0) * (x - 1.0));
		return (beta * 0.25) * (x * x - 1.0) * (x * x - 1.0);
	}
	virtual double Sd(double x) const override
	{
		if (x < -1.0)
			return beta * (2.0 * x + 2.0);
		if (x > 1.0)
			return beta * (2.0 * x - 2.0);
		return beta * (x * x * x - x);
	}
};

class Periodic : public Action
{
  public:
	double beta;
	explicit Periodic(double beta) : beta(beta) {}
	virtual double S(double x) const override
	{
		double scale = 2.0 * M_PI;
		return -beta * std::cos(x * scale);
	}
	virtual double Sd(double x) const override
	{
		double scale = 2.0 * M_PI;
		return beta * std::sin(x * scale) * scale;
	}
	virtual double normalize(double x) const override
	{
		while (x > 0.5)
			x -= 1.0;
		while (x < -0.5)
			x += 1.0;
		return x;
	}
};

class PeriodicWell : public Action
{
  public:
	double beta, a;
	explicit PeriodicWell(double beta = 1.0, double a = 1.0) : beta(beta), a(a)
	{}

	virtual double S(double x) const override
	{
		return beta * (0.5 * x * x - a * std::cos(2 * M_PI * x));
	}
	virtual double Sd(double x) const override
	{
		return beta * (x + a * 2 * M_PI * std::sin(2 * M_PI * x));
	}
};

std::vector<double> runMarkov(const Action &action, int count, double eps,
                              int seed)
{
	if (seed == -1)
		seed = std::random_device()();
	xoshiro256 rng(seed);
	std::normal_distribution<double> dist(0, std::sqrt(2.0));

	double x = 0.0; // cold start
	std::vector<double> data;
	data.reserve(count);
	for (int i = -count / 2; i < count; ++i)
	{
		double eta = dist(rng);

		double f = eps * action.Sd(x) + std::sqrt(eps) * eta;
		x -= f;
		x = action.normalize(x);
		if (i >= 0)
			data.push_back(x);
	}
	return data;
}

int main(int argc, char **argv)
{
	CLI::App app{"1D toy model to test ideas about Markov processes"};

	std::string action_name = "";
	int count = 1000000;
	std::string filename = "";
	bool force = false;
	bool rescaling = true;
	int seed = -1;

	double beta_min = 1.0;
	double beta_max = 1.0;
	double beta_count = 1;

	double eps_min = 0.02;
	double eps_max = 0.2;
	int eps_count = 20;

	app.add_option("--action", action_name);
	app.add_option("--count", count);
	app.add_option("--filename", filename);
	app.add_flag("--force", force);
	app.add_option("--rescaling", rescaling);
	app.add_option("--seed", seed, "PRNG seed. -1 for unpredictable");

	app.add_option("--eps_min", eps_min);
	app.add_option("--eps_max", eps_max);
	app.add_option("--eps_count", eps_count);

	app.add_option("--beta_min", beta_min);
	app.add_option("--beta_max", beta_max);
	app.add_option("--beta_count", beta_count);

	CLI11_PARSE(app, argc, argv);

	DataFile file;
	if (filename != "")
		file = DataFile::create(filename, force);
	int i = 1;

	auto plot = Gnuplot();

	for (int beta_i = 0; beta_i < beta_count; ++beta_i)
	{
		double beta =
		    beta_min + (beta_max - beta_min) * beta_i / (beta_count - 1);
		if (beta_count == 1)
			beta = beta_min;

		std::vector<double> xs, ys;
		for (int eps_i = 0; eps_i < eps_count; ++eps_i)
		{
			double eps =
			    eps_min + (eps_max - eps_min) * eps_i / (eps_count - 1);
			if (eps_count == 1)
				eps = eps_min;

			double delta = eps;
			if (rescaling)
				delta /= beta;

			std::unique_ptr<Action> action;
			if (action_name == "harmonic")
				action = std::make_unique<Harmonic>();
			else if (action_name == "double_well")
				action = std::make_unique<DoubleWell>(beta);
			else if (action_name == "periodic")
				action = std::make_unique<Periodic>(beta);
			else
				assert(false);

			fmt::print("eps = {}, beta = {}\n", eps, beta);
			auto data = runMarkov(*action, count, delta, seed);

			if (file)
			{
				auto set = file.createData(fmt::format("chain_{}", i++), data);
				set.setAttribute("eps", eps);
				set.setAttribute("beta", beta);
			}

			if (eps_count == 1)
				xs.push_back(beta);
			else
				xs.push_back(eps);
			ys.push_back(variance(data));

			// only a single parameter-set -> show a histogram
			if (eps_count == 1 && beta_count == 1)
			{
				int binCount = 100;
				auto h = Histogram(data, binCount);
				double s = 0.0;
				for (i = 0; i < binCount; ++i)
				{
					auto x = 0.5 * (h.mins[i] + h.maxs[i]);
					s += std::exp(-action->S(x));
					fmt::print("{}, {} {}\n", x, -action->S(x),
					           std::exp(-action->S(x)));
				}
				auto f = [&](double x) {
					return data.size() / s * std::exp(-action->S(x));
				};
				Gnuplot().plotHistogram(h).plotFunction(f, h.mins.front(),
				                                        h.maxs.back());
			}
		}
		if (eps_count > 1 || beta_count > 1)
			plot.plotData(xs, ys, fmt::format("beta = {}", beta));
	}
}