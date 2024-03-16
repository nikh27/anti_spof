from django.db import models

# Create your models here.
from django.db import models
from django.utils import timezone


class User(models.Model):
    name = models.CharField(max_length=100)
    roll = models.CharField(max_length=20)

    def __str__(self):
        return self.name

class Attendance(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    time = models.TimeField(default=timezone.now)

    def __str__(self):
        return f"{self.user.name} - {self.date}"


        
