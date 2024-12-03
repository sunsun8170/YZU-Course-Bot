"""Main script for the CAPTCHA course selection bot.

This script initializes and runs the course selection bot by loading user-defined 
course configurations from a YAML file. It includes error handling for file 
loading and YAML parsing issues, ensuring the process runs smoothly.
"""
import sys
import yaml

from src.bot import CourseBot
import src.utils as utils

try:
  with open("course_list.yaml", "r", encoding="utf-8") as f:
    usr_course_list = yaml.safe_load(f)["course_list"]
except FileNotFoundError as e:
  print(f"[ 找不到檔案 ] 請確認 course_list.yaml 是否存在!\n詳細資訊: {e}")
  sys.exit(0)
except yaml.YAMLError as e:
  print(f"[ YAML 解析錯誤 ] 請確認 course_list.yaml 中語法是否正確!\n詳細資訊: {e}")
  sys.exit(0)
except Exception as e:
  print(f"[ 未知的錯誤 ]\n詳細資訊: {e}")
  sys.exit(0)

utils.enable_signal_handler()

bot = CourseBot(usr_course_list)
bot.run()
