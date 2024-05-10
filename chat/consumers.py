# chat/consumers.py
import json
import torch
import random
from nltk_utils import bag_of_words, tokenize
from django.conf import settings 
from channels.generic.websocket import AsyncWebsocketConsumer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
        self.room_group_name = "chat_%s" % self.room_name

        # Join room group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json["message"]

        sentence = tokenize(message)
        X = bag_of_words(sentence, settings.ALL_WORDS)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = settings.MODEL(X)
        _, predicted = torch.max(output, dim=1)

        tag = settings.TAGS[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]


        result = "I do not understand..."

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    result = f"{random.choice(intent['responses'])}"

        # Send message to room group
        await self.channel_layer.group_send(self.room_group_name, {"type": "chat_message", "message": message})
        await self.channel_layer.group_send(self.room_group_name, {"type": "chat_bot_message", "message": result})

    # Receive message from room group
    async def chat_message(self, event):
        message = event["message"]

        # Send message to WebSocket
        await self.send(text_data=json.dumps({"message": message}))

    async def chat_bot_message(self, event):
        message = event["message"]

        # Send message to WebSocket
        await self.send(text_data=json.dumps({"bot_message": message}))
