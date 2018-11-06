[2018-11-06 21:58:12] load stop words ...
Traceback (most recent call last):
  File "./src/1_basic_feature.py", line 26, in <module>
    with open(config.STOP_WORDS_FILE, 'r', encoding="utf-8") as f:
FileNotFoundError: [Errno 2] No such file or directory: './data/stop_words.txt'
