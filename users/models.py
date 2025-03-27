from django.db import models
from django.contrib.auth.models import User

class ChatMemory(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    memory = models.JSONField(default=list)  # List of {"user": "...", "bot": "..."} dictionaries

    def __str__(self):
        return f"ChatMemory for {self.user.username}"