"""Script for managing and running the course bot.

This script handles the initialization and execution of the course bot, which automates 
the process of managing course selections based on user-defined configurations.
"""
from functools import wraps
import getpass
import io
import os
import re
import sys
import time
from typing import Any, Callable, Dict, Tuple

from bs4 import BeautifulSoup
from PIL import Image
import requests
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.configs import BotConfigs
import src.utils as utils


def handle_exceptions(wait: int = 0.5) -> Callable[[Callable], Callable]:
  """A decorator to handle exceptions and automatically retry the decorated 
  function.

  This decorator catches exceptions during the execution of the decorated 
  function. If a network-related exception occurs, the function will be retried 
  after a specified wait time. For other types of exceptions, the program will 
  log the error or terminate.

  Args:
    wait: 
      The amount of time, in seconds, to wait before retrying 
      the function after a network-related error. Defaults to 0.5 seconds.

  Returns:
    Callable: 
      A decorator that wraps the provided function with exception 
      handling and retry logic.
  """

  def decorator(func: Callable) -> Callable:

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
      while True:
        try:
          return func(self, *args, **kwargs)
        except requests.RequestException as e:
          self._logger.error(
              f"[ 網路異常 ] 嘗試連線中! 詳細資訊: {e}",
              exc_info=BotConfigs.DEBUG_MODE,
          )
          time.sleep(wait)
        except Exception as e:
          self._logger.critical(f"[ 未知的錯誤 ]\n詳細資訊: {e}", exc_info=True)
          sys.exit(0)

    return wrapper

  return decorator


class CourseBot:
  """
  A bot designed for automating course selection at Yuan Ze University.

  This class provides functionalities to log in, select courses, and handle 
  related operations seamlessly. It integrates with an Optical Character Recognition 
  (OCR) model for CAPTCHA decoding and uses a logger for action tracking. 

  Attributes:
    _account: 
      The username used for logging into the course selection system.
    _password: 
      The password associated with the account for authentication.
    _logger: 
      An instance of a logger to record the bot's actions, statuses, 
      and error messages.
    _processor:
      A processor for handling image tokenization and text decoding using 
      the TrOCR model.
    _model:
      A vision-based text recognition model used for CAPTCHA processing.
    _session:
      An HTTP session for maintaining persistent connections during requests.
    _select_payload:
      A dictionary containing the payload used during course selection 
      requests.
    _usr_course_list:
      A list of course IDs or names that the user intends to select.
    _courses_db:
      A dictionary containing details of available courses, including course IDs,
      descriptions, and prerequisites.
    _boosted:
      A flag indicating whether the bot should employ an aggressive strategy for
      course selection, such as faster retries.
  """

  def __init__(
      self,
      usr_course_list: list,
  ):
    """Initialize the course bot.

    This constructor sets up the course bot with the necessary configurations 
    and prepares it to perform operations such as logging in, selecting courses, 
    and managing course data.

    Args:
      usr_course_list:
        A list of course identifiers and names that the bot will attempt 
        to select. Each entry represents a course the user intends to enroll in.
    """
    self._account = ""
    self._password = ""
    self._logger = self._init_logger()
    self._processor, self._model = self._init_model()
    self._session = self._init_session()
    self._select_payload: Dict[str, Any] = {}
    self._usr_course_list = usr_course_list
    self._courses_db: Dict[str, Any] = {}
    self._boosted = True  # for more explanation, see the dynamic_delay method

  def _init_model(self) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Load the TrOCR model and processor.

    This method initializes the TrOCR processor and vision-based text recognition 
    model using the specified pretrained model name or path. The processor is 
    used for tokenizing inputs and decoding outputs, while the model performs 
    OCR tasks.

    Args:
      pretrained_model_name:
        The name of the pretrained TrOCR model to be loaded from a known source 
        (e.g., Hugging Face model hub).
      pretrained_model_path:
        The local path to a pretrained TrOCR model checkpoint. This is used if 
        the model is not loaded directly by name.

    Returns:
      tuple: 
        A tuple containing:
        - `processor`: The processor for tokenizing and decoding text.
        - `model`: The OCR model for image-to-text tasks.

    Raises:
      Exception: 
        If the model or processor fails to load due to incorrect name/path or 
        compatibility issues.
    """
    try:
      device = "cuda" if torch.cuda.is_available() else "cpu"
      processor = TrOCRProcessor.from_pretrained(
          BotConfigs.PRETRAINED_MODEL_NAME,
          clean_up_tokenization_spaces=True,
          char_level=True,
      )
      model = VisionEncoderDecoderModel.from_pretrained(
          BotConfigs.PRETRAINED_MODEL_PATH).to(device)
      self._logger.info(f"[ 模型載入成功 ] 使用 {BotConfigs.PRETRAINED_MODEL_NAME} 模型!")

      return processor, model

    except Exception as e:
      self._logger.critical(
          f"[ 模型載入失敗 ]\n詳細資訊: {e}",
          exc_info=BotConfigs.DEBUG_MODE,
      )

      sys.exit(0)

  def _init_logger(self) -> object:
    """Initialize and configure the logger object.

    This method creates a logger instance to record messages and events, such as 
    errors, warnings, and informational messages during the execution of the bot. 
    The logger can be configured to output to the console and a file.

    Returns:
      logging.Logger: 
        A configured logger object for tracking the bot's activity.
    """
    logger = utils.logger(BotConfigs.LOG_DIR)

    return logger

  @staticmethod
  def _init_session() -> requests.Session:
    """Initialize and configure an HTTP requests session.

    This method creates a session object to manage and persist settings across 
    multiple HTTP requests. The session can handle cookies, headers, and other 
    configurations for consistent interactions with web services.

    Returns:
      requests.Session: 
        A configured session object for making HTTP requests.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent":
            "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0",
        "Accept":
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8",
        "Accept-Language":
            "en-US,en;q=0.5",
        "Referer":
            BotConfigs.LOGIN_URL_REFERER,
        "Accept-Encoding":
            "gzip, deflate, br, zstd",
        "Upgrade-Insecure-Requests":
            "1",
    })

    return session

  def _init_login_payload(self, **kwargs) -> Dict[str, str]:
    """Initialize the login payload with required and additional values.

    This method prepares a dictionary containing the default login credentials 
    and any additional parameters provided. The payload is used for authenticating 
    requests during the login process.

    Args:
      kwargs:
        Optional key-value pairs to include in the payload, which may 
        override default values or add extra fields.

    Returns:
      dict: 
        A dictionary representing the complete login payload to be 
        submitted during the authentication process.
    """
    payload = {
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "Txt_User": self._account,
        "Txt_Password": self._password,
        "btnOK": "確定"
    }
    payload.update(kwargs)

    return payload

  def _init_select_payload(self, dept_id: str, **kwargs) -> Dict[str, str]:
    """Initialize the course selection payload with required and optional values.

    This method creates a dictionary containing the default values necessary 
    for selecting a course, such as the department ID, while allowing additional 
    key-value pairs to be included for customization.

    Args:
      dept_id: 
        The ID of the department to which the course belongs. This value 
        is mandatory for the selection process.
      kwargs:
        Optional key-value pairs to include in the payload, allowing for 
        additional or overridden parameters.

    Returns:
      dict: 
        A dictionary representing the complete payload for the course 
        selection process.
    """
    payload = {
        "__EVENTARGUMENT": "",
        "__LASTFOCUS": "",
        "__VIEWSTATEENCRYPTED": "",
        "Hidden1": "",
        "Hid_SchTime": "",
        "DPL_DeptName": dept_id,
        "DPL_Degree": "6"
    }
    payload.update(kwargs)

    return payload

  def _get_captcha_text(self, image: Image.Image) -> str:
    """Perform Optical Character Recognition (OCR) on a CAPTCHA image.

    This method uses the TrOCR processor and model to process the input CAPTCHA 
    image and extract the recognized text. The expected output is a string of 
    exactly 4 characters, corresponding to the text in the CAPTCHA.

    Args:
      image:
        The input image containing the CAPTCHA to be processed.

    Returns:
      str: 
        The recognized text extracted from the CAPTCHA, consisting of exactly 
        4 characters.

    Raises:
      Exception: 
        If the OCR process encounters an error, such as failing to recognize the
        CAPTCHA or issues with the input image or model.
    """
    try:
      device = "cuda" if torch.cuda.is_available() else "cpu"
      pixel_values = self._processor(
          image,
          return_tensors="pt",
      ).pixel_values.to(device)
      generated_ids = self._model.generate(pixel_values)

      return self._processor.batch_decode(
          generated_ids,
          skip_special_tokens=True,
      )[0]

    except Exception as e:
      self._logger.critical(
          f"[ 模型辨識錯誤 ] 在辨識驗證碼文字時發生了未知的錯誤!\n詳細資訊: {e}",
          exc_info=BotConfigs.DEBUG_MODE,
      )

      sys.exit(0)

  @staticmethod
  def _clean_alert_msg(response_text: str) -> str:
    """Clean the alert message from the response text.

    This method processes the response text from the course selection system to 
    remove unwanted characters commonly found in JavaScript alert messages, such 
    as '(r)', '(c)', '(c.)', '\\n', and other formatting issues. The result is a 
    cleaner, more readable alert message.

    Args:
      response_text:
        The raw response text containing the alert message, which may include 
        unwanted characters or formatting.

    Returns:
      str: 
        The cleaned alert message, with extraneous characters and formatting 
        removed.
    """
    alert_msg = re.search(r"alert\(['\"](.*?)['\"]\)", response_text).group(1)
    alert_msg = re.sub(r"[()\.\r\\nrc]", "", alert_msg)

    return alert_msg

  @staticmethod
  def _clear_terminal() -> None:
    """Clear the terminal screen.
    
    This function clears the terminal or console screen, providing a clean 
    slate for further output. It can be useful in interactive or script-based 
    environments where you want to refresh the display. Depending on the 
    operating system, the function uses the appropriate command (e.g., `cls` for
    Windows or `clear` for Unix-like systems) to clear the screen.
    """
    os.system("cls" if os.name == "nt" else "clear")

  @handle_exceptions()
  def _login(self) -> bool:
    """Log into the course selection system using account credentials and CAPTCHA.

    This method attempts to log in to the course selection system by providing 
    the necessary account credentials (username and password) along with a 
    CAPTCHA code for verification. If the CAPTCHA is recognized correctly, 
    the login is considered successful.

    Returns:
      bool: 
        True if the login is successful (i.e., correct CAPTCHA is provided and the 
        system accepts the credentials), False otherwise (i.e., incorrect account, 
        password, or CAPTCHA).
    """
    while True:
      self._session.cookies.clear()

      captcha_response = self._session.get(
          BotConfigs.CAPTCHA_URL,
          stream=True,
          timeout=BotConfigs.REQUEST_TIMEOUT,
      )
      captcha_response.raise_for_status()

      captcha_data = io.BytesIO(captcha_response.content)
      captcha_img = Image.open(captcha_data).convert("RGB")
      captcha_text = self._get_captcha_text(captcha_img)

      login_response = self._session.get(BotConfigs.LOGIN_URL)
      login_response.raise_for_status()

      if "選課系統尚未開放" in login_response.text:
        self._logger.info("[ 選課系統尚未開放 ]")
        continue

      parser = BeautifulSoup(login_response.text, "lxml")

      login_payload = self._init_login_payload(
          __VIEWSTATE=parser.select_one("#__VIEWSTATE")["value"],
          __VIEWSTATEGENERATOR=parser.select_one("#__VIEWSTATEGENERATOR")
          ["value"],
          __EVENTVALIDATION=parser.select_one("#__EVENTVALIDATION")["value"],
          DPL_SelCosType=parser.select_one(
              "#DPL_SelCosType option:not([value='00'])")["value"],
          Txt_CheckCode=captcha_text,
      )

      result = self._session.post(
          BotConfigs.LOGIN_URL,
          data=login_payload,
          timeout=BotConfigs.REQUEST_TIMEOUT,
      )
      result.raise_for_status()

      is_logged_in = self._handle_login_result(result.text)
      if isinstance(is_logged_in, bool):
        return is_logged_in

  def _handle_login_result(self, response_text: str) -> bool:
    """Handle the login result and display appropriate messages based on the 
    response.

    This method processes the response text from the login request to determine 
    whether the login attempt was successful. If the login is successful, a success 
    message is displayed. Otherwise, an error message is shown to indicate the
    failure.

    Args:
      response_text:
        The response text received from the login request, which contains 
        information about the login result.

    Returns:
      bool: 
        True if the login is successful (i.e., the response indicates a successful 
        login), False otherwise (i.e., the response indicates failure or error).
    """
    succeeded_message = "parent.location ='SelCurr.aspx?Culture=zh-tw'"
    retry_message = "驗證碼錯誤"

    if succeeded_message in response_text:
      self._logger.info("[ 登入成功 ]")
      return True

    elif retry_message in response_text:
      self._logger.warning("[ 登入失敗 ] 驗證碼錯誤， 重新嘗試中...")
      return None

    else:
      alert_msg = self._clean_alert_msg(response_text)
      self._logger.error(f"[ 登入失敗 ] {alert_msg}")
      input("請按任意鍵繼續...")
      self._clear_terminal()
      return False

  @handle_exceptions()
  def _verify_usr_course_list(self) -> list:
    """Verify the user's course list.

    This method checks each course in the user's list to ensure the validity of 
    the department ID and course ID. If any course has an invalid department or 
    course ID, it will be ignored, and the program will continue processing the 
    remaining valid courses.

    Returns:
      list: 
        A list of verified courses that have valid department and course IDs.
    """
    self._logger.info("[ 開始檢查選課清單 ]")
    verified_courses = []

    for option in self._usr_course_list:
      option = option.replace(" ", "")
      usr_dept_id, *rest = option.split(",")
      usr_course_id = ",".join(rest)

      html = self._session.get(
          BotConfigs.COURSE_LIST_URL,
          timeout=BotConfigs.REQUEST_TIMEOUT,
      )
      html.raise_for_status()
      parser = BeautifulSoup(html.text, "lxml")

      if "異常登入" in html.text:
        self._logger.critical("[ 帳號被阻擋 ] 已被相關單位偵測到頻繁搶課!")
        sys.exit(0)

      sys_dept_id = parser.select(
          f"#DPL_DeptName option[value='{usr_dept_id}']")
      if not sys_dept_id:
        self._logger.warning(
            f"[ 已忽略選項 ] 選項 {option} 錯誤, 系所代號 {usr_dept_id} 不存在!")
        continue
      sys_dept_name = sys_dept_id[0].text

      select_payload = self._init_select_payload(
          usr_dept_id,
          __EVENTTARGET="DPL_Degree",
          __VIEWSTATE=parser.select_one("#__VIEWSTATE")["value"],
          __VIEWSTATEGENERATOR=parser.select_one("#__VIEWSTATEGENERATOR")
          ["value"],
          __EVENTVALIDATION=parser.select_one("#__EVENTVALIDATION")["value"],
      )

      html = self._session.post(
          BotConfigs.COURSE_LIST_URL,
          data=select_payload,
          timeout=BotConfigs.REQUEST_TIMEOUT,
      )
      html.raise_for_status()
      parser = BeautifulSoup(html.text, "lxml")

      sys_course_id = parser.select(
          f"#CosListTable input[name*='{usr_course_id}']")

      if not sys_course_id:
        self._logger.warning(
            f"[ 已忽略選項 ] 選項 {option} 錯誤, 系所 {sys_dept_name} 查無 {usr_course_id} 課程!",
        )
        continue

      self._select_payload[usr_dept_id] = select_payload

      sys_course_id = sys_course_id[0]
      sys_course_name = sys_course_id.attrs["name"].split(" ")[-1]
      sys_course_info = f"{usr_course_id} {sys_course_name}"
      self._courses_db[usr_course_id] = {
          "info": sys_course_info,
          "mUrl": sys_course_id.attrs["name"]
      }

      verified_courses.append(option)

    if not verified_courses:
      self._logger.error(
          "[ 選課清單為空 ] 在忽略所有錯誤的選項後, 選課清單為空! 請重新檢查 course_list.json!",)
      sys.exit(0)

    self._logger.info("[ 選課清單檢查完成 ]")

    return verified_courses

  @handle_exceptions()
  def _select_courses(self, verified_usr_course_list: list) -> None:
    """Automatically select courses from the verified course list.

    This method iterates through the provided list of verified courses and
    attempts to automatically select each course. The course selection process 
    will only proceed for courses that have passed verification.

    Args:
      verified_usr_course_list:
        A list of verified courses that are valid and ready for selection.
    """
    time_stamp = time.time()

    self._logger.info("[ 開始搶課 ]")

    while verified_usr_course_list:
      for option in verified_usr_course_list.copy():
        usr_dept_id, *rest = option.split(",")
        usr_course_id = ",".join(rest)

        html = self._session.post(
            BotConfigs.COURSE_LIST_URL,
            data=self._select_payload[usr_dept_id],
            timeout=BotConfigs.REQUEST_TIMEOUT,
        )
        html.raise_for_status()

        parser = BeautifulSoup(html.text, "lxml")
        select_payload = self._init_select_payload(
            usr_dept_id,
            __EVENTTARGET="",
            __VIEWSTATE=parser.select_one("#__VIEWSTATE")["value"],
            __VIEWSTATEGENERATOR=parser.select_one("#__VIEWSTATEGENERATOR")
            ["value"],
            __EVENTVALIDATION=parser.select_one("#__EVENTVALIDATION")["value"],
        )
        m_url = self._courses_db[usr_course_id]["mUrl"]
        select_payload.update({f"{m_url}.x": "0", f"{m_url}.y": "0"})

        response = self._session.post(
            BotConfigs.COURSE_LIST_URL,
            data=select_payload,
            timeout=BotConfigs.REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        html = self._session.get(
            f"{BotConfigs.COURSE_SELECT_URL}{m_url} ,B,",
            timeout=BotConfigs.REQUEST_TIMEOUT,
        )
        html.raise_for_status()

        self._handle_select_courses_result(
            html.text,
            verified_usr_course_list,
            option,
            usr_course_id,
        )

        time.sleep(self.dynamic_delay(time_stamp))

    self._logger.info("[ 結束搶課 ] 選課清單中所有可搶課程已搶畢!")

  def _handle_select_courses_result(
      self,
      response_text: str,
      verified_usr_course_list: list,
      option: str,
      usr_course_id: str,
  ) -> None:
    """
    Handle the result of selecting courses and display messages accordingly.

    This method processes the response from the course selection system after 
    attempting to select a course. It checks whether the selection was successful 
    and displays appropriate messages based on the result. If the course selection 
    was unsuccessful, it will provide information on why it failed. Additionally, 
    it handles any error or failure scenarios during the course selection process.

    Args:
      response_text:
        The response text returned from the course selection request, 
        containing information about the success or failure of the selection.
      verified_usr_course_list:
        A list of verified courses that are being processed for selection.
      option:
        The specific course option or selection being processed.
      usr_course_id:
        The unique ID of the course being selected.
    """
    alert_msg = self._clean_alert_msg(response_text)
    detailed_info = f"{self._courses_db[usr_course_id]['info']} \t{alert_msg}"

    succeeded_messages = ["加選訊息", "已選過", "完成加選"]
    retry_messages = ["開放外系生可加選", "人數已達上限"]
    failed_messages = ["異常查詢課程資訊", "斷線", "逾時", "logged off"]
    critical_messages = ["異常登入"]

    if any(msg in response_text for msg in retry_messages):
      self._logger.info(f"[ 持續搶課中 ] {detailed_info}")
      return
    elif any(msg in response_text for msg in succeeded_messages):
      self._logger.info(f"[ 已成功加選 ] {detailed_info}")
      verified_usr_course_list.remove(option)
      return
    elif any(msg in response_text for msg in failed_messages):
      self._logger.error(f"[ 重新連線中 ] 已由選課系統登出!")
      self._login()
      return
    elif any(msg in response_text for msg in critical_messages):
      self._logger.critical(f"[ 帳號被阻擋 ] {detailed_info}")
      sys.exit(0)
    else:
      self._logger.warning(f"[ 已忽略選項 ] {detailed_info}")
      verified_usr_course_list.remove(option)

  def dynamic_delay(self, time_stamp: float) -> int:
    """Set the frequency of course selection dynamically.

    This method adjusts the frequency of course selection based on the provided 
    timestamp. The timestamp indicates when the course selection 
    process starts, and the frequency may be modified dynamically based on different
    conditions. For example, the delay time between course selections can be
    reduced in the first few seconds to increase the chances of selecting
    courses. After a certain period, the delay time is increased to avoid being 
    banned by the system.

    Args:
      time_stamp:
        The timestamp (in seconds) representing the start time of the course 
        selection process. This value is used to calculate the appropriate 
        frequency or interval for subsequent actions.

    Returns:
      int: 
        The delay time in seconds between course selection actions. If the 
        selection happens within the first 5 seconds, no delay is applied. 
        After 5 seconds, a delay of 3 seconds is applied (return 3).
    """
    if not self._boosted:
      # not recommended to reduce the delay time, you might get banned lol
      return 3

    elapsed_time = (time.time() - time_stamp) % 60
    if elapsed_time < 5:
      # greedily select courses in the first 5 seconds, no delay time set
      # you can select around 50 to nearly 60 times in the first 5 seconds
      # not recommended to extend more then 5 seconds, you might get banned
      return 0
    else:
      # after being greedy, set the boosted flag to False,
      # then set the delay time to 3 seconds to avoid being banned
      self._boosted = False
      return 3

  def run(self) -> None:
    """Begin the automated course selection process.

    This method initiates the process of logging into the course selection system, 
    selecting the courses specified by the user, and handling the various 
    operations required to complete the selection. It manages the entire flow 
    from login to course selection, ensuring that the system's behavior (e.g., 
    CAPTCHA handling, login checks, and course selection) is correctly addressed 
    during execution.

    The process will involve:
      - Logging in with provided account credentials.
      - Handling CAPTCHA for authentication.
      - Verifying the list of courses to be selected.
      - Dynamically adjusting the frequency of course selections to optimize the process.
      - Selecting courses based on the validated list.
    """
    self._clear_terminal()

    # login
    while True:
      print("YZU Course Bot v2.0")
      print("Please enter your YZU Portal account and password. "
            "Your password will be hidden while typing.")
      self._account = input("Account: ")

      if self._account == "bark":
        print(""" 
            |\_/|                  
            | @ @   Woof! Woof!
            |   <>              _  
            |  _/\------____ ((| |))
            |               `--' |   
        ____|_       ___|   |___.' 
        /_/_____/____/_______|
        """)
        input("\"Stupid Humans!\", doggy said ...")
        self._clear_terminal()
        continue
      elif self._account == "exit":
        sys.exit(0)

      self._password = getpass.getpass(prompt="Password: ")

      if self._login():
        break

    # verify user's course list
    verified_courses = self._verify_usr_course_list()

    # automatically select courses
    self._select_courses(verified_courses)
