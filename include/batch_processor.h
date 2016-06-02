#pragma once
#include "thread_processor.hpp"
namespace ParallelDo {
/**
 * Use this to post a series of jobs to run in parallel and then wait for
 * them to complete in the scheduling thread
 */
class BatchTracker : boost::noncopyable
{
	public:
		BatchTracker(ThreadProcessor *threadProcessorp):
			number_of_jobs_total(0), number_of_jobs_complete(0),
			cond_(), mutex(), threadProcessorp_(threadProcessorp) { }
		virtual ~BatchTracker() { };

		/**
		 * post a function to run in parallel
		 * @func a function that takes no arguments to run in parallel, use
		 * a boost::bind to schedule with arguments
		 */
		void post(boost::function<void ()> func) {
			threadProcessorp_->post(boost::bind(&BatchTracker::wrap, this, func));
			incJobCount();
		}
		void postWorkList(boost::function<void ()> func) {
			threadProcessorp_->post(boost::bind(&BatchTracker::wrap, this, func));
			incJobCount();
		}

		/**
		 * @param seconds max number of seconds to wait for all jobs to
		 * complete
		 */
		bool wait_until_done(time_t seconds = 0) {
			if (number_of_jobs_complete == number_of_jobs_total)
				return true;
			return wait_until_done_locked(seconds);
		}

		/**
		 * call this before re-using a batch tracker after calling
		 * wait_until_done
		 */
		void reset() {
			number_of_jobs_total = number_of_jobs_complete = 0;
		}

		/**
		 * returns number of time post was called since instansiation or
		 * reset() called
		 */
		int scheduled() const {
			return number_of_jobs_total;
		}

		/**
		 * number of jobs that have completed since instansiation or reset()
		 * called
		 */
		int complete() const {
			return number_of_jobs_complete;
		}

	private:
		int incJobCount() {
			return number_of_jobs_total++;

		}

		void done() {
			boost::mutex::scoped_lock lock(mutex);
			number_of_jobs_complete++;
			if (number_of_jobs_complete == number_of_jobs_total)
				cond_.notify_one();
		}

		void wrap(boost::function<void ()> func) {
			func();
			done();
		}

		bool wait_until_done_locked(time_t max_seconds) {
			boost::mutex::scoped_lock lock(mutex);
			time_t start, now = start = time(NULL);
			do {
				cond_.wait(mutex);
				if (number_of_jobs_complete != number_of_jobs_total) {
					now = time(NULL);
				}
			} while (number_of_jobs_complete != number_of_jobs_total &&
					(max_seconds == 0 || ((now - start) < max_seconds)));
			return (number_of_jobs_complete == number_of_jobs_total);
		}

		volatile int number_of_jobs_total;
		volatile int number_of_jobs_complete;
		boost::condition cond_;
		boost::mutex mutex;
		ThreadProcessor *threadProcessorp_;
};
}
