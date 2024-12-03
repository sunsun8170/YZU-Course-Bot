"""This module provides utility functions for the course bot.

The utilities include functionalities for:
- Logging: Setup and management of log files.
- Signal Handling: Enabling graceful interruption of the program execution.
"""
import logging
import os
import signal
import sys
import time


def logger(save_path: str = "logs", max_log_files: int = 10) -> None:
  """Create and configure a logger object for the bot.

  This function initializes a logger with settings for both log files and console
  logging. It includes log file rotation, retaining only the most recent 
  `max_log_files` files to optimize disk space usage.

  Args:
    save_path:
      The directory path where log files will be saved. Defaults to "./logs".
    max_log_files:
      The maximum number of log files to retain in the `save_path` directory. 
      Older files beyond this limit are deleted, starting with the oldest. 
      Defaults to 10.

  Returns:
    logging.Logger:
      A logger instance configured with both file and console handlers to 
      capture and display log messages.
  """
  os.makedirs(save_path, exist_ok=True)

  log_files = [
      os.path.join(save_path, f)
      for f in os.listdir(save_path)
      if f.endswith('_yzuCourseBot_log.txt')
  ]
  log_files.sort(key=os.path.getctime, reverse=True)

  for file in log_files[max_log_files:]:
    os.remove(file)

  log_filename = os.path.join(
      save_path,
      time.strftime("%Y-%m-%d_%H-%M-%S") + "_yzuCourseBot_log.txt",
  )
  log_format = f"%(asctime)-30s%(levelname)-10s%(message)s"
  logging.basicConfig(
      level=logging.INFO,
      format=log_format,
      handlers=[
          logging.FileHandler(log_filename),
          logging.StreamHandler(),
      ],
  )

  return logging.getLogger(__name__)


def enable_signal_handler() -> None:
  """Enable a signal handler to manage interruptions gracefully.

  This function sets up a signal handler to intercept the `SIGINT` signal 
  (typically triggered by `Ctrl+C`). When an interruption occurs, it prompts 
  the user to either continue or terminate the program.
  """

  def signal_handler(sig, frame) -> None:
    """Handle the SIGINT signal (interruption).

    This nested function is called when a `SIGINT` signal is received. It 
    interacts with the user to decide whether to continue execution or exit.

    Args:
      sig:
        The signal number indicating the type of signal received (e.g., SIGINT).
      frame:
        The current stack frame at the time the signal was received.
    """
    while True:
      answer = input("繼續執行? [Y/n]: ").lower()
      if answer == 'y' or answer == "":
        return
      elif answer == 'n':
        print("正在退出...")
        sys.exit(0)
      else:
        print("無效輸入！")

  signal.signal(signal.SIGINT, signal_handler)
