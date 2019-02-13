#ifndef UTIL_STOPWATCH_H
#define UTIL_STOPWATCH_H

#include <chrono>

namespace util {

/** Simple Stopwatch class for performance measruements */
class Stopwatch
{
	using Clock = std::chrono::steady_clock;

	bool running_ = false;
	typename Clock::time_point last_; // valid if running
	typename Clock::duration dur_ = Clock::duration::zero();

  public:
	Stopwatch() {}

	Stopwatch &start()
	{
		if (!running_)
		{
			running_ = true;
			last_ = Clock::now();
		}
		return *this;
	}

	Stopwatch &stop()
	{
		if (running_)
		{
			running_ = false;
			dur_ += Clock::now() - last_;
		}
		return *this;
	}

	double secs() const
	{
		typename Clock::duration dur = dur_;
		if (running_)
			dur += Clock::now() - last_;
		return std::chrono::duration_cast<std::chrono::microseconds>(dur)
		           .count() *
		       1.0e-6;
	}
};

} // namespace util

#endif
