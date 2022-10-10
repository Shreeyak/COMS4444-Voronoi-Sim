# Voronoi Pygame Sim

Much faster simulator for the voronoi game. Primary features:
- Speed: 1000 days in ~36 sec (default player)
- Identical player interface: Just replace `players/<gr>_player.py` file with the file from official sim's 
  `players/<gr>_player.py` file. No changes required.
- Debug code: A breakpoint in the code will pause the simulation, allowing analysis of the current game state
  with visuals.
- Video export: Export a video of the game, if needed. Just add a `--video` flag.

The simulator uses pygame for the visualization. It also has a few under-the-hood optimizations to make it run faster.

![fast](images/demo-sim.gif)

### Launch:

```shell
python main.py --days 100 --spawn 5 -p1 d -p2 1 -p3 8 -p4 5 
```

Usage:
```
usage: main.py [-h] [--map_size MAP_SIZE] [--no_gui] [--days DAYS] [--spawn SPAWN] 
    [--player1 PLAYER1] [--player2 PLAYER2] [--player3 PLAYER3] [--player4 PLAYER4] 
    [--fps FPS] [--timeout TIMEOUT] [--seed SEED] [--out_video]

COMS 4444: Voronoi

optional arguments:
  -h, --help            show this help message and exit
  --map_size MAP_SIZE, -m MAP_SIZE
                        Size of the map in km
  --no_gui, -g          Disable GUI
  --days DAYS, -d DAYS  Total number of days
  --spawn SPAWN         Number of days after which a new unit spawns at the homebase
  --player1 PLAYER1, -p1 PLAYER1
                        Specify player 1 out of 4. Integer equal to player id.
  --player2 PLAYER2, -p2 PLAYER2
                        Specify player 2 out of 4. Integer equal to player id.
  --player3 PLAYER3, -p3 PLAYER3
                        Specify player 3 out of 4. Integer equal to player id.
  --player4 PLAYER4, -p4 PLAYER4
                        Specify player 4 out of 4. Integer equal to player id.
  --fps FPS, -f FPS     Max speed of simulation
  --timeout TIMEOUT, -t TIMEOUT
                        Timeout for each players execution. 0 to disable
  --seed SEED, -s SEED  Seed used by random number generator. 0 to disable.
  --out_video, -o       If passed, save a video of the run. Slows down the simulation 2x.
```

In the simulator
```
Key bindings:  
  - Esc: Quit the game.   
  - R: Reset game
```

### Install

Project was created with Python 3.9. Install dependencies with:

```shell
pip install -r requirements.txt
```

### Coordinate System

We use an X-right, Y-down coordinate system. (same as original sim) 
In order to make computation easy, our coordinate system follows from numpy indexing (Origin: Top-left)  

Note: Y-axes represents columns. So a unit at `location[10, 30]` is at `cell[30, 10]`.

# Voronoi Interactive Sim

![](images/demo.gif)
<p align="center">100x100 grid showing occupancy</p>

A simulator to visualize the voronoi game with different unit placements.  
Click to add units

Key bindings:  
  - Esc: Quit the game.  
  - 1-4: Select player 0-3  
  - R: Reset game
  - K: Kill isolated units  



#### Launch:

```shell
python main_interactive.py [-m <map_size>]
```

## Features

1. Blazing Fast! Uses KDTree to find nearest points for each cell.
   In a 100x100 grid with 100 units each player,
   the initial occupancy is computed in 170ms.
2. Variable Grid Size - Visualize strategies on a smaller map


# Dev thoughts

- TODO: Add player color next to player name
- TODO: TA sim slows down as it progresses. Port optimizations from here.
- TODO: Use Voronoi diagram - map each player to unit. Must be able to find a node/voronoi cell given
  a unit ID. Create a graph from this map. Use to identify isolated units and build occ map.


Ambition:
- Pause game and enter interactive mode. Add units and see what will happen.

Interactive Mode:
- TODO: Remove units with click (shift-click or change mode with key)
- TODO: (Low priority) Click and drag to move units

- TODO: Remi - replace SVG graphics with an image container
       VIDEO: https://www.reddit.com/r/RemiGUI/comments/9skkag/streaming_images_from_a_webcam/

 
#### Game State 
Stores a list of units, compute occupancy map, killed units. Compute new unit pos based on move
commands for each unit. Occupancy map and killed units can be brute force (more efficient).
Needs to maintain exact coords and unique ID of each unit. This is passed to players.
      
Exposed to Player: 
   - List of units (player, exact pos, id). 
   - Occupancy Map. 
   - Curr/total score.
   - Curr/total days.
   - (removed to match interface) History (units, move commands, occ map) - easier motion tracking
   - (removed to match interface) Move units. 
  
#### GUI 
Main code that is launched and handles all interfaces.  
Allows: start/stop/reset game, enable interactive mode.  
Does: Initialize player code,  log errors.  
