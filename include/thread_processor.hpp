// vim: set noexpandtab
#ifndef THREAD_PROCESSOR_HPP
#define THREAD_PROCESSOR_HPP

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/utility.hpp>

namespace ParallelDo {
class ThreadProcessor {
	public:
		ThreadProcessor(int max_wait_for_job = 500, int number_of_worker_threads = 0, int jobs_per_worker=5):
			io_mutex_(), cond_(),
			initialized_(false), actors_(),
			number_messages_(0), done_(false), message_list_(),
			max_wait_for_job_(max_wait_for_job),
			number_of_worker_threads_(number_of_worker_threads), jobs_per_worker_(jobs_per_worker),exit_func_(NULL) {
				start_workers();
			}
		typedef boost::function<void ()> work_t;

		void set_exit_func(ThreadProcessor::work_t exit_func) {
			exit_func_ = exit_func;
		}

		void set_done() {
			boost::mutex::scoped_lock lock(io_mutex_);
			done_ = true;
		}

		~ThreadProcessor() {
			set_done();
			cond_.notify_all();
			//actors_.interrupt_all();
			actors_.join_all();
		}

		/**
		 * you can use this to schedule a task to run if you don't care to
		 * wait for it to complete
		 *
		 * func should be a function that takes no arguments
		 */
		void post(work_t func) {
			boost::mutex::scoped_lock lock(io_mutex_);
			message_list_.push_back(func);
			number_messages_++;
			lock.unlock();
			cond_.notify_one();
		}
		void postWorkList(std::list<work_t> &worklist) {
			boost::mutex::scoped_lock lock(io_mutex_);
			int size = worklist.size();
			message_list_.splice(message_list_.end(),worklist );
			number_messages_+=size;
			lock.unlock();
			cond_.notify_one();
		}

		int queued() const {	//this is intensionally not locked
			return number_messages_;
		}
	private:
		bool getJobs(std::list<work_t> &jobs) {
			boost::mutex::scoped_lock lock(io_mutex_);
			while (false == done_ && queued() == 0) {
				boost::system_time tAbsoluteTime = boost::get_system_time() + boost::posix_time::milliseconds(max_wait_for_job_);
				cond_.timed_wait(io_mutex_, tAbsoluteTime);
			}
			int available = queued();
			if (available == 0)
				return false;
			int jobs_to_get;
			if (available <= jobs_per_worker_)
				jobs_to_get = 1;
			else
				jobs_to_get = jobs_per_worker_;
			std::list<work_t>::iterator itr = message_list_.begin();
			int i;
			for (i=0; i < jobs_to_get; i++) {
				++itr;
			}

			jobs.splice(jobs.end(), message_list_, message_list_.begin(), itr);
			number_messages_-=i;
			if (number_messages_> 0) {
				lock.unlock();
				cond_.notify_one();
			}
			return true;
		}

		void worker(int worker_id __attribute__((unused))) {

			std::list<work_t> myWorkList;
			while (!done_) {
				work_t job;
				bool success = getJobs(myWorkList);
				if (false == success)
					break;
				while (myWorkList.size() > 0) {
					work_t job = myWorkList.front();
					myWorkList.pop_front();
					job();
				}
				if (exit_func_)
					exit_func_();
			}
		}

		void start_workers() {
			if (false == initialized_) {
				int number_of_threads =
					0 == number_of_worker_threads_ ? boost::thread::hardware_concurrency() : number_of_worker_threads_;
				for (int i = 0; i < number_of_threads; i++) {
					actors_.create_thread(boost::bind(&ThreadProcessor::worker, this, i));
				}
				initialized_ = true;
			}
		}
		boost::mutex io_mutex_;
		boost::condition cond_;

		bool initialized_;
		boost::thread_group actors_;

		volatile int number_messages_;
		volatile bool done_;

		std::list<work_t> message_list_;
		int max_wait_for_job_;
		int number_of_worker_threads_;
		int jobs_per_worker_;
		work_t exit_func_;
	public:
};

}
#endif
