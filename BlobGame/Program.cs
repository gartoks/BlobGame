﻿// Entry point.

using BlobGame;
using BlobGame.App;
using BlobGame.Game.GameModes;
using SimpleGL;
using SimpleGL.Util;
using System.Diagnostics;
using System.Text;

bool socketMode = false;
int numParallelGames = 1;
bool useSeparateThreads = false;
int port = 0;
int seed = 0;
string gameModeKey = "Classic";

DEBUGStuff.DEBUG_PrintNPatchJson();

Log.OnLog += (msg, type) => Console.WriteLine(msg);
Log.OnLog += (msg, type) => Debug.WriteLine(msg);


try {
    if (args.Length > 0) {
        for (int i = 0; i < args.Length; i++) {
            if (!args[i].StartsWith("--"))
                continue;

            if (args[i] == "--debug")
                GameApplication.DRAW_DEBUG = true;

            if (args[i] == "--sockets" &&
                i + 4 < args.Length &&
                int.TryParse(args[i + 1], out numParallelGames) &&
                bool.TryParse(args[i + 2], out useSeparateThreads) &&
                int.TryParse(args[i + 3], out port) &&
                int.TryParse(args[i + 4], out seed) &&
                IGameMode.GameModeTypes.ContainsKey(args[i + 5])
                ) {
                socketMode = true;
                gameModeKey = args[i + 5];
            }
        }
    }

    if (socketMode) {
        SocketApplication.Initialize(numParallelGames, useSeparateThreads, port, seed, gameModeKey);
        SocketApplication.Start();
    } else {
        Application.Initialize(new GameApplication());
        Application.Start();
    }
} catch (Exception e) {
    StringBuilder sb = new StringBuilder();

    Exception? ex = e;
    while (ex != null) {
        sb.AppendLine(ex.ToString());
        ex = ex.InnerException;
    }

    sb.AppendLine();
    sb.AppendLine(e.StackTrace);

    File.WriteAllText("error.log", sb.ToString());

    throw;
}