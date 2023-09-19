class Player:
    def play(self):
        print(colored("The player is playing cricket.\n\n",'blue'))

class Batsman(Player):
    def play(self):
        print(colored("The batsman is batting.\n\n",'blue'))

class Bowler(Player):
    def play(self):
        print(colored("The bowler is bowling.\n\n",'blue'))
batsman = Batsman()
bowler = Bowler()
from termcolor import colored
print(colored("\t\t\t\tCRICKET PLAYER \n\n\n",'red'))
# Calling the function name with object 

batsman.play()
bowler.play()