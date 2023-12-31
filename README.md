# Toasted!
A totally unique and original concept and definitely not a clone of Suika Game, Toasted! is a physics-based food combination puzzle game. Earn your crust and craft the king of breakfast: Fairy Bread! (Or maybe it's a toaster pastry?)

## Features
- Brand New game mode with new food shapes!
- Tetris-style Hold function for your convenience!
- Alternate food evolution paths!
- Fully controllable with socket! Even multiple games at once!
- Cute breakfast themed gameplay!
- Relaxing music to vibe to!
- Custom theme compatibility!
- Linux support! (Although you have to build the game yourself)
- Mac support! (Technically, we didn't have mac to test it on)

## License

The game is under the GNU GPLv3 License - see the [LICENSE.md](LICENSE.md) file for details. This means any derivative work needs to also use this license, as well as open source their code.
If you want to not do that, please contact us and we will work something out.

## Custom Themes
The game is designed to to be (almost) fully re-themeable!
Swap out textures, sounds, music, make your own tutorials, etc.
Inside the "Resources" folder are files with the file extensions ".theme". Open them as a zip Archive (using 7zip or similar) and simply swap and/or edit files.
If files are swapped, they need to have the exact same name as the files in the current theme.

Example:
Want to swap out "title_logo.png"?
Simply replace it with your own file called "title_logo.png".

If there are multiple .theme files in the Resources folder, the game will recognize this and you can swap the theme in the "Settings" menu at runtime.

If some files are missing, they won't load (obviously).
In the best case the texture simply isn't rendered, or the sound not played. In the worst case the game crashes.
For texture files you should also keep an eye out for the aspect ratio of image files. If width/height are changed these textures may appear stretched ingame.

## Socket Control
The game is fully controllable via sockets and TCP/IP. You can also start and play several games at once using socket control without any of them being drawn. Very efficient!.
If you want to start a game in socket mode and actually see what is happening, simply select the "Socket Controller" when starting a new game and enter the port your server runs on.
If you want the parallel/no-gui version start the .exe with the following arguments

--sockets [NumberOfParallelGames] [UseSeparatedThreads] [Port] [Seed] [GameMode]

- [NumberOfParallelGames] Determines how many games should be started at once.
- [UseSeparatedThreads] can take either 'true' or 'false'. If set to true, the game attempts to run the game in parallel, ideally using multiple CPU cores.
- [Port] The port your controller-server runs on. The game will try to connect it using this port.
- [Seed] The seed your game uses for spawning objects. Same seed = same order of objects. If multiple games are run the seed increments from game to game. So game 1 has seed X, game 2 seed X+1, etc.
- [GameMode] The game mode you want your games to be. Currently there are 2 game modes: "Classic" and "Toasted" (capitalization matters).

### Sending Instructions
A instruction to the game is 6 bytes in length.
The first 4 bytes represent a "t value" in [0, 1]. It controls the positioning of your cursor/object dopper above the arena. 0 is all the way left, 1 all the way right. It is a 32bit floating point number that complies with IEC 60559:1989 (IEEE 754).

Byte 5 is a flag that tells the game to drop the current object if possible. Only the first bit is taken into account. So 00000000 tells the game not to drop the object this frane. 00000001 tells the game to drop the object this frame.

Byte 6 is a flag that tells the game to hold the current object if possible. Only the first bit is taken into account. So 00000000 tells the game not to hold the object this frane. 00000001 tells the game to hold the object this frame.

### Receiving date from the game
The data received from the game each frame potentially differs in length from frame to frame.

The first 4 bytes are a signed 32bit integer that represents the number of objects currently in play in the arena.

The following bytes are in chunks and represent data for each object. There will be number-of-objects-in-play chunks.
Object data consists of (in that order):
- 32bit floating point number (4 btyes) representing the x coordinate of the object. It complies with IEC 60559:1989 (IEEE 754).
- 32bit floating point number (4 btyes) representing the y coordinate of the object. It complies with IEC 60559:1989 (IEEE 754).
- 32bit signed integer (4 bytes) representing the type of the object.

After the object data the next 4 bytes are a signed 32bit integer of type of the object dropped next.

The next 4 bytes are a signed 32bit integer of type of the object that is being drop next, after the current object.

The next 4 bytes are a signed 32bit integer of type of the object that is currently being held. If that functionality is not supported by the game mode, it is always 0.

The next 4 bytes are a signed 32bit integer representing your current score.

The next 4 bytes are a signed 32bit integer representing the index of your current game. This can be used to differentiate the game when playing multiple games at once.

The next 1 byte is a flag indicating wether the object can be dropped this frame. Only the first bit is taken into account. So 00000000 means that the object can not be dropped. 00000001 means that the object can be dropped.

The next 1 byte is a flag indicating if the game is over. Only the first bit is taken into account. So 00000000 the game is still running. 00000001 means that the game is over.

The next 4 bytes are a signed 32bit integer that represents the length of the game mode in bytes.
The next bytes are the name of the game mode as a string encoded with UTF8.

## Credits

### Artwork
- Ixzyl

### Sound
- Ixzyl
- Fibi

### Code
- gartoks
- Robotino

## Contact / Contributing
If you have questions, please reach out to gartoks or Ixzyl on Discord.

## Changelog

## Known Bugs
- Switching to fullscreen sometimes crashes the game.
- Monitor switching is currently disabled.
- re-theming currently requires restarting the game

## TODO:
- [ ] The smallest two Blobs allow for three Blobs to be combined into one if they simultaneously collide.
- [ ] Add a custom physics engine (maybe)
- [ ] Feature: shoving the arena

## Acknowledgments
Inspired by SUIKA for the game, and Melba Toast for the theme.