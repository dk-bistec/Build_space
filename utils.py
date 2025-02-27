import os
from dotenv import load_dotenv, find_dotenv

# tload env diel                                                                                                                    
def load_env():
    _ = load_dotenv(find_dotenv())



# load groc api key
def get_groc_api_key():
    load_env()
    groc_api_key = os.getenv("GROQ_API_KEY")
    return groc_api_key


































'''
# break to 100
# don't break in the middle 
def pretty_print_result(result):
  parsed_result = []
  for line in result.split('\n'):
      if len(line) > 80:
          words = line.split(' ')
          new_line = ''
          for word in words:
              if len(new_line) + len(word) + 1 > 100:
                  parsed_result.append(new_line)
                  new_line = word
              else:
                  if new_line == '':
                      new_line = word
                  else:
                      new_line += ' ' + word
          parsed_result.append(new_line)
      else:
          parsed_result.append(line)
  return "\n".join(parsed_result)'''
