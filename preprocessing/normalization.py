# IMPORTS
import re


def normalize(string):
    """
        Normalize given string.

        Input:
            - string: string

        Output:
            - normalized string
    """

    reg = r"MENTION[A-Za-z0-9_]*"
    string = remove_emojis(string)
    string = remove_hashtags(string)
    string = remove_mentions(string)
    string = remove_links(string)
    string = remove_word(string, reg)
    return string


def remove_emojis(string):
    """
        Remove emojis.

            Input:
                - string: string

            Output:
                - string without emojis
    """
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', string)


def remove_hashtags(string):
    """
        Remove hashtags (#).

            Input:
                - string: string

            Output:
                - string without hashtags
    """
    clean_tweet = re.sub(r"#[A-Za-z0-9_]+", "", string)
    clean_tweet = re.sub(r' +', ' ', clean_tweet)
    return clean_tweet.strip()


def remove_mentions(string):
    """
        Remove mentions (@).

            Input:
                - string: string

            Output:
                - string without mentions
    """
    clean_tweet = re.sub(r"@[A-Za-z0-9_]+", "", string)
    clean_tweet = re.sub(r' +', ' ', clean_tweet)
    return clean_tweet.strip()


def remove_links(string):
    """
        Remove links.

            Input:
                - string: string

            Output:
                - string without links
    """
    clean_tweet = re.sub(r"https?://\S+", "", string)
    clean_tweet = re.sub(r' +', ' ', clean_tweet)
    return clean_tweet.strip()


def remove_word(string, regex):
    """
        Remove specific regex.

            Input:
                - string: string
                - regex: string

            Output:
                - string without regex
    """
    clean_tweet = re.sub(regex, "", string)
    clean_tweet = re.sub(r' +', ' ', clean_tweet)
    return clean_tweet.strip()


def set_to_string(data):
    """
        Convert given data into string.

            Input:
                - data

            Output:
                - data as string
    """
    return str(data)


def normalize_list(data):
    """
        Iterate over list and convert nested list to list.

            Input:
                - data: nested list

            Output:
                - list
    """
    return_list = []
    for element in data:
        if type(element) is str:
            return_list.append(element)
        if type(element) is list:
            if len(element) > 0:
                return_list.append(element[0])
            else:
                return_list.append(str(element))
    return return_list
