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

## Install

Project was created with Python 3.9. Install dependencies with:

```shell
pip install -r requirements.txt
```

## Coordinate System

We use an X-right, Y-down coordinate system.  
In order to make computation easy, our coordinate system follows from numpy indexing (Origin: Top-left)  

Note: Y-axes represents columns. So a unit at `location[10, 30]` is at `cell[30, 10]`.


## Features

1. Blazing Fast! Uses KDTree to find nearest points for each cell.
   In a 100x100 grid with 100 units each player,
   the initial occupancy is computed in 170ms. 
   ![fast](images/speed_100x100_400pts.png)
   <p align="center">Occupancy Grid: 100x100 grid with 400 random units. 170ms to compute.</p>

2. Variable Grid Size - Visualize different strategies easily


## Dev thoughts

- TODO: Use Voronoi diagram - map each player to unit. Must be able to find a node/voronoi cell given
  a unit ID. Create a graph from this map. Use to identify isolated units and build occ map.
- TODO: BFS search from the above graph to find all nodes of a given player.

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
   - History (units, move commands, occ map) - easier motion tracking
   - Move units. 
 
Internal: 
   - Unit cell occupancy.
   - Connected map (kill units). 
   - Reset (calc game state from scratch, called after units move/are killed).

 
#### GUI 
Main code that is launched and handles all interfaces.  
Allows: start/stop/reset game, enable interactive mode.  
Does: Initialize player code,  log errors.  
