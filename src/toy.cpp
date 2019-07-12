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
			return beta * ((x + 1.0) * (x + 1.0) - 0.25);
		if (x > 1.0)
			return beta * ((x - 1.0) * (x - 1.0) - 0.25);
		return beta * (0.25 * x * x * x * x - 0.5 * x * x);
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

std::vector<double> runMarkov(const Action &action, int count, double eps)
{
	xoshiro256 rng((std::random_device()()));
	std::normal_distribution<double> dist(0, std::sqrt(2.0));

	double x = 0.0; // cold start
	std::vector<double> data;
	data.reserve(count);
	for (int i = -count / 2; i < count; ++i)
	{
		double eta = dist(rng);

		double f = eps * action.Sd(x) + std::sqrt(eps) * eta;
		x -= f;
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
	bool rescaling = true;

	double beta_min = 1.0;
	double beta_max = 1.0;
	double beta_count = 1;

	double eps_min = 0.02;
	double eps_max = 0.2;
	int eps_count = 20;

	app.add_option("--action", action_name);
	app.add_option("--count", count);
	app.add_option("--filename", filename);
	app.add_option("--rescaling", rescaling);

	app.add_option("--eps_min", eps_min);
	app.add_option("--eps_max", eps_max);
	app.add_option("--eps_count", eps_count);

	app.add_option("--beta_min", beta_min);
	app.add_option("--beta_max", beta_max);
	app.add_option("--beta_count", beta_count);

	CLI11_PARSE(app, argc, argv);

	DataFile file;
	if (filename != "")
		file = DataFile::create(filename);
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

			double delta = eps;
			if (rescaling)
				delta /= beta;

			std::unique_ptr<Action> action;
			if (action_name == "harmonic")
				action = std::make_unique<Harmonic>();
			else if (action_name == "double_well")
				action = std::make_unique<DoubleWell>(beta);
			else
				assert(false);

			fmt::print("eps = {}\n", eps);
			auto data = runMarkov(*action, count, delta);

			if (file)
			{
				auto set = file.createData(fmt::format("chain_{}", i++), data);
				set.setAttribute("eps", eps);
				set.setAttribute("beta", beta);
			}

			xs.push_back(eps);
			ys.push_back(variance(data));
		}
		plot.plotData(xs, ys, fmt::format("beta = {}", beta));
	}
}
