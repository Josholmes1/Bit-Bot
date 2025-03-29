import pygame

def play_buy_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("buy_alert.mp3")
    pygame.mixer.music.play()
    input("Press Enter after hearing the Buy sound...")

def play_sell_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("sell_alert.mp3")
    pygame.mixer.music.play()
    input("Press Enter after hearing the Sell sound...")

print("🔊 Testing Buy Sound...")
play_buy_sound()

print("🔴 Testing Sell Sound...")
play_sell_sound()
