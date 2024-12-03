"""This module defines configuration constants for the CAPTCHA course selection 
bot.

The `BotConfigs` class contains all the necessary constants, such as URLs for the 
system endpoints, paths for saving logs and models, and general operational 
settings. It ensures that configurations remain immutable and well-organized.
"""
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class BotConfigs:
  """Configuration class for the course bot.

  This class contains constants used for the bot's operation, including URLs, 
  model settings, and general configurations.

  Attributes:
    LOGIN_URL_REFERER:
      The URL for the login page referer.
    LOGIN_URL:
      The URL for the login endpoint.
    CAPTCHA_URL:
      The URL for fetching CAPTCHA images.
    COURSE_LIST_URL:
      The URL for fetching the list of available courses.
    COURSE_SELECT_URL:
      The URL for selecting a course.
    PRETRAINED_MODEL_NAME:
      The name of the pretrained OCR model.
    PRETRAINED_MODEL_PATH:
      The file path to the pretrained OCR model.
    REQUEST_TIMEOUT:
      Timeout for HTTP requests, in seconds.
    DEBUG_MODE:
      Enable or disable detailed error logging.
    LOG_DIR:
      Directory for saving log files.
  """
  # url configs
  LOGIN_URL_REFERER: Final = "https://isdna1.yzu.edu.tw/Cnstdsel/default.aspx"
  LOGIN_URL: Final = "https://isdna1.yzu.edu.tw/CnStdSel/Index.aspx"
  CAPTCHA_URL: Final = "https://isdna1.yzu.edu.tw/CnStdSel/SelRandomImage.aspx"
  COURSE_LIST_URL: Final = "https://isdna1.yzu.edu.tw/CnStdSel/SelCurr/CosList.aspx"
  COURSE_SELECT_URL: Final = "https://isdna1.yzu.edu.tw/CnStdSel/SelCurr/CurrMainTrans.aspx?mSelType=SelCos&mUrl="
  # model configs
  PRETRAINED_MODEL_NAME: Final = "microsoft/trocr-small-printed"
  PRETRAINED_MODEL_PATH: Final = "./model"
  # general configs
  REQUEST_TIMEOUT: Final = 5
  DEBUG_MODE: Final = False  # set to True to enable detailed error messages
  LOG_DIR: Final = "./logs"
