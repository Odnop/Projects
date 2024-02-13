
import random 

# Defining Fair roulette

class FairRoulette():
    def __init__(self):
        self.pockets = []
        for i in range(1,37):
            self.pockets.append(i)
        self.ball = None
        self.pocketOdds = len(self.pockets) - 1
    def spin(self):
        self.ball = random.choice(self.pockets)
    def betPocket(self, pocket, amt):
        if str(pocket) == str(self.ball):
            return amt*self.pocketOdds
        else: return -amt
    def __str__(self):
        return 'Fair Roulette'  
FairRoulette()

# Roulette function

def playRoulette(game, numSpins, pocket, bet, toPrint):
    totPocket = 0
    for i in range(numSpins):
        game.spin()
        totPocket += game.betPocket(pocket, bet)
    if toPrint:
        print(numSpins, 'spins of', game)
        print('Expected return betting', pocket, '=', \
              str(100*totPocket/numSpins) + '%\n')
    return (totPocket/numSpins)

game = FairRoulette()
for numSpins in (100,10000000):
    for i in range(3):
        playRoulette(game, numSpins, 30, 1, True)
        
game = FairRoulette()
for numSpins in (100, 1000000):
    for i in range(3):
        playRoulette(game, numSpins, 30, 1, True)
        
# European roulette includes one zero

class EuRoulette(FairRoulette):
    def __init__(self):
        FairRoulette.__init__(self)
        self.pockets.append('0')
    def __str__(self):
        return 'European Roulette'

Eu_game = EuRoulette()
for numSpins in (100, 1000, 10000, 100000, 1000000):
    for i in range(3):
        playRoulette(Eu_game, numSpins, 30, 1, True)

# American roulette includes two zeroes
    
class AmRoulette(EuRoulette):
    def __init__(self):
        EuRoulette.__init__(self)
        self.pockets.append('00')
    def __str__(self):
        return 'American Roulette'
    
    
Am_game = AmRoulette()
for numSpins in (100, 1000, 10000, 100000, 1000000):
    for i in range(3):
        playRoulette(Am_game, numSpins, 30, 1, True)
        
# Plot the results

import matplotlib.pyplot as plt

results_Am = {}
for numSpins in [100, 1000, 10000, 100000, 1000000, 10000000]:
    Am_game_results = []
    for i in range(3):
        Am_result = playRoulette(Am_game, numSpins, 30, 1, True)
        Am_game_results.append(Am_result)
    results_Am[numSpins] = Am_game_results
    
results_Eu = {}
for numSpins in [100, 1000, 10000, 100000, 1000000, 10000000]:
    Eu_game_results = []
    for i in range(3):
        Eu_result = playRoulette(Eu_game, numSpins, 30, 1, True)
        Eu_game_results.append(Eu_result)
    results_Eu[numSpins] = Eu_game_results
    

results_Fair = {}
for numSpins in [100, 1000, 10000, 100000, 1000000, 10000000]:
    Fair_game_results = []
    for i in range(3):
        Fair_result = playRoulette(FairRoulette(), numSpins, 30, 1, True)
        Fair_game_results.append(Fair_result)
    results_Fair[numSpins] = Fair_game_results

# Plot for Fair Roulette
for numSpins, Fair_game_results in results_Fair.items():
    plt.plot(Fair_game_results, label=f'{numSpins} spins')

plt.xlabel('Trial')
plt.ylabel('Result')
plt.title('Fair Roulette Results')
plt.legend()
plt.show()

# Plot for European Roulette
for numSpins, Eu_game_results in results_Eu.items():
    plt.plot(Eu_game_results, label=f'{numSpins} spins')

plt.xlabel('Trial')
plt.ylabel('Result')
plt.title('European Roulette Results')
plt.legend()
plt.show()

# Plot for American Roulette
for numSpins, Am_game_results in results_Am.items():
    plt.plot(Am_game_results, label=f'{numSpins} spins')

plt.xlabel('Trial')
plt.ylabel('Result')
plt.title('American Roulette Results')
plt.legend()
plt.show()


