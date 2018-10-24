import re
import urllib.request
import urllib.parse

url_google = 'http://translate.google.cn'
reg_text = re.compile(r'(?<=TRANSLATED_TEXT=).*?;')
user_agent = r'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) ' \
                 r'Chrome/44.0.2403.157 Safari/537.36'

def translateGoogle(text, f='en', t='zh-cn'):
    values = {'hl': 'zh-cn', 'ie': 'utf-8', 'text': text, 'langpair': '%s|%s' % (f, t)}
    value = urllib.parse.urlencode(values)
    req = urllib.request.Request(url_google + '?' + value)
    req.add_header('User-Agent', user_agent)
    response = urllib.request.urlopen(req)
    content = response.read().decode('utf-8')
    data = reg_text.search(content)
    result = data.group(0).strip(';').strip('\'')
    print(result)

# text="translate corpus content"
text="234"
translateGoogle(text)
