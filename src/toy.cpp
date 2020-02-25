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

// parameters of a general 3-step scheme
struct IntegratorParams
{
	int steps = -1;
	double k1 = 0, k2 = 0, k3 = 0, k4 = 0, k5 = 0, k6 = 0;
	double c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0;
};

struct MarkovResults
{
	std::vector<double> configs;
	std::vector<double> moment2;
	std::vector<double> moment4;
};

MarkovResults runMarkov(const Action &action, int64_t count, size_t spacing,
                        double eps, int seed, const IntegratorParams &params)
{
	if (seed == -1)
		seed = std::random_device()();
	xoshiro256 rng(seed);
	std::normal_distribution<double> dist(0, std::sqrt(2.0));

	double x = 0.0; // cold start
	MarkovResults result;
	if ((int64_t)count * 8 * 3 > 128 * 1024 * 1024)
		throw std::runtime_error(
		    "requested chain larger than 128MiB. Probably a mistake");
	result.configs.reserve(count);
	result.moment2.reserve(count);
	result.moment4.reserve(count);
	double seps = std::sqrt(eps);
	for (int64_t i = -count / 4; i < count; ++i)
	{
		double moment2 = 0;
		double moment4 = 0;
		for (size_t iter = 0; iter < spacing; ++iter)
		{
			double eta1 = dist(rng);
			double eta2 = dist(rng);
			double eta3 = dist(rng);

			double f1 =
			    params.k1 * eps * action.Sd(x) + params.c1 * seps * eta1;
			double f2 = params.k2 * eps * action.Sd(x) +
			            params.k3 * eps * action.Sd(x - f1) +
			            params.c2 * seps * eta1 + params.c3 * seps * eta2;
			double f3 = params.k4 * eps * action.Sd(x) +
			            params.k5 * eps * action.Sd(x - f1) +
			            params.k6 * eps * action.Sd(x - f2) +
			            params.c4 * seps * eta1 + params.c5 * seps * eta2 +
			            params.c6 * seps * eta3;
			if (params.steps == 1)
				x -= f1;
			else if (params.steps == 2)
				x -= f2;
			else if (params.steps == 3)
				x -= f3;
			else
				assert(false);
			x = action.normalize(x);

			moment2 += x * x;
			moment4 += (x * x) * (x * x);
		}

		if (i >= 0)
		{
			result.configs.push_back(x);
			result.moment2.push_back(moment2 / spacing);
			result.moment4.push_back(moment4 / spacing);
		}
	}
	return result;
}

int main(int argc, char **argv)
{
	CLI::App app{"1D toy model to test ideas about Markov processes"};

	std::string action_name = "";
	int count = 200000;
	int spacing = 5;
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
	app.add_option("--spacing", spacing);
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

	std::string scheme;
	app.add_option("--scheme", scheme);

	CLI11_PARSE(app, argc, argv);

	IntegratorParams params;
	if (scheme == "euler")
	{
		params.steps = 1;
		params.k1 = 1;
		params.c1 = 1;
	}
	else if (scheme == "bf")
	{
		params.steps = 2;
		params.k1 = 1;
		params.c1 = 1;
		params.k2 = 0.5;
		params.k3 = 0.5;
		params.c2 = 1;
	}
	else if (scheme == "torrero")
	{
		params.steps = 2;
		params.k1 = 0.08578643762690485;
		params.c1 = 0.2928932188134524;
		params.k2 = 0;
		params.k3 = 1;
		params.c2 = 1;
	}
	else if (scheme == "burger")
	{
		// minimal k1 without any additional noise
		params.steps = 3;

		params.k1 = 0.0413944133642607;
		params.k2 = 0.0564549335355903;
		params.k3 = 0.122177861418491;
		params.k4 = 0.25;
		params.k5 = 0;
		params.k6 = 0.75;

		params.c1 = 0.203456170622227;
		params.c2 = 0.422649730810374;
		params.c3 = 0;
		params.c4 = 1;
		params.c5 = 0;
		params.c6 = 0;
	}
	else if (scheme == "burger_b")
	{
		// minimal k1 with only c5 as extra noise
		params.steps = 3;

		params.k1 = 0.0226731615535121;
		params.k2 = 0;
		params.k3 = 0.264297739604484;
		params.k4 = 0.25;
		params.k5 = 0;
		params.k6 = 0.75;

		params.c1 = 0.150576098878647;
		params.c2 = 0.514098958960708;
		params.c3 = 0;
		params.c4 = 0.905433078636425;
		params.c5 = 0.424489034146897;
		params.c6 = 0;
	}
	else
	{
		fmt::print("unknown scheme: '{}'", scheme);
		return -1;
	}

	DataFile file;
	if (filename != "")
	{
		file = DataFile::create(filename, force);
		file.setAttribute("action", action_name);
		file.setAttribute("scheme", scheme);
		file.makeGroup("configs");
		file.makeGroup("moment2");
		file.makeGroup("moment4");
	}
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

			// auto data = runMarkov(*action, count, delta, seed, params);
			auto result =
			    runMarkov(*action, count, spacing, delta, seed, params);
			fmt::print(
			    "eps = {}, beta = {}, mom2 = {}, mom4 = {}, ac(mom2) = {}\n",
			    eps, beta, mean(result.moment2), mean(result.moment4),
			    util::correlationTime(result.moment2));

			if (file)
			{
				{
					auto set = file.createData(
					    fmt::format("configs/chain_{}", i++), result.configs);
					set.setAttribute("eps", eps);
					set.setAttribute("beta", beta);
				}
				{
					auto set = file.createData(
					    fmt::format("moment2/chain_{}", i++), result.moment2);
					set.setAttribute("eps", eps);
					set.setAttribute("beta", beta);
				}
				{
					auto set = file.createData(
					    fmt::format("moment4/chain_{}", i++), result.moment4);
					set.setAttribute("eps", eps);
					set.setAttribute("beta", beta);
				}
			}

			if (eps_count == 1)
				xs.push_back(beta);
			else
				xs.push_back(eps);
			ys.push_back(mean(result.moment2));

			// only a single parameter-set -> show a histogram
			if (eps_count == 1 && beta_count == 1)
			{
				int binCount = 100;
				auto h = Histogram(result.configs, binCount);
				double s = 0.0;
				for (i = 0; i < binCount; ++i)
				{
					auto x = 0.5 * (h.mins[i] + h.maxs[i]);
					s += std::exp(-action->S(x));
					fmt::print("{}, {} {}\n", x, -action->S(x),
					           std::exp(-action->S(x)));
				}
				auto f = [&](double x) {
					return result.configs.size() / s * std::exp(-action->S(x));
				};
				Gnuplot().plotHistogram(h).plotFunction(f, h.mins.front(),
				                                        h.maxs.back());
			}
		}
		if (eps_count > 1 || beta_count > 1)
			plot.plotData(xs, ys, fmt::format("beta = {}", beta));
	}

	if (file)
		file.close();
}
