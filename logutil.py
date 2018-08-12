import logging
import os
import time

def initlogger():
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    if not os.path.exists("./log"):
        os.makedirs("./log")
    handler = logging.FileHandler('./log/info-%s.log' % (time.strftime('%Y%m%d', time.localtime(time.time()))),
                                  encoding='utf-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    log.addHandler(handler)
    return log

logger = initlogger()