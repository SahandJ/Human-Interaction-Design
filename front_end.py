import re
import discord
from Chatbot import *


client = discord.Client()
@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')


@client.event
async def on_message(message):
    global answer
    if message.author.name != client.user.name and not message.content.startswith('$'):
        await  client.send_typing(message.channel)
        print(message.author.name)
        print(message.content)
        message_split =re.findall(r"[A-Za-z@#]+|\S", message.content)
        message_split_len=len(message_split)

        print(message_split_len >= 3 or '?' in message_split )
        if (message_split_len >= 3 or '?' in message_split):
            try:
                topic,answer = brain.ask_ai_v2(message_split)
            except TypeError:
                await client.send_message(message.channel,
                                      'Har ingen flere svar til dette. Spør gjerne om noe nytt')
            

            cleanr = re.compile('<.*?>')
            answer=re.sub(cleanr,'',answer)
            d={'&aelig;':"æ",'&oslash;':"ø","&aring;":'å'}
            for key, value in d.items():
                if key in answer:
                    answer = re.sub(key,value,answer)
            # answer=re.sub("&aring;","å",answer)
            # answer=re.sub("&oslash;","ø",answer)
            # answer=re.sub("&aelig;","æ",answer)
            await client.send_message(message.channel,
                                      'Machine Learning: Tema {} vil snakke om er: "{}"'.format(message.author.name,topic),)
            await client.send_message(message.channel,
                                      'Hentet Data Fra ung.no: {}'.format(answer),tts=True)
        else:
            await client.send_message(message.channel,"Question to short")
            brain.db.clear_prob_to_answer()
    if message.content.startswith('$clear'): 
        await client.send_message(message.channel, "New Person, cleared memory..")
        brain.db.clear_prob_to_answer()




import logging

logging.basicConfig(level=logging.INFO)

token = 'MzU5MzMxNDc1NzQ2NzE3NzA4.DKFdCA.8D1Ytw_HK21Y3_DKYMiKT4BYxKM'
def main():
   client.run(token)


if __name__ == '__main__':

    brain = AI(chating=True)
    main()