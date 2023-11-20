﻿// Entry point.

using BlobGame;
using BlobGame.Game.GameModes;
using BlobGame.Util;
using System.Diagnostics;
using System.Text;

string usage_text = $"""
usage: 
    {System.Reflection.Assembly.GetExecutingAssembly().Location}
    OR
    {System.Reflection.Assembly.GetExecutingAssembly().Location} --sockets [use_threads] [seed] [port] [gamemode]
""";

bool socketMode = false;
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
                Application.DRAW_DEBUG = true;

            if (args[i] == "--sockets" &&
                i + 4 < args.Length &&
                bool.TryParse(args[i + 1], out useSeparateThreads) &&
                int.TryParse(args[i + 2], out seed) &&
                int.TryParse(args[i + 3], out port) &&
                IGameMode.GameModeTypes.ContainsKey(args[i + 4])
                ) {
                socketMode = true;
                gameModeKey = args[i + 4];
                i += 4;
            }
            else{
                Console.WriteLine($"Unknown argument: {args[i]}");
                Console.WriteLine(usage_text);
                Environment.Exit(1);
            }
        }
    }

    if (socketMode) {
        SocketApplication.Initialize(useSeparateThreads, seed, port, gameModeKey);
        SocketApplication.Start();
    } else {
        Application.Initialize();
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