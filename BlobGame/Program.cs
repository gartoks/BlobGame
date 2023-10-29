// Entry point.

using BlobGame;

bool socketMode = false;
int numParallelGames = 1;
bool useSeparateThreads = false;
int seed = 0;
int port = 0;

if (args.Length > 0) {
    for (int i = 0; i < args.Length; i++) {
        if (!args[i].StartsWith("--"))
            continue;

        if (args[i] == "--debug")
            Application.DRAW_DEBUG = true;

        if (args[i] == "--sockets" &&
            i + 4 < args.Length &&
            int.TryParse(args[i + 1], out numParallelGames) &&
            bool.TryParse(args[i + 2], out useSeparateThreads) &&
            int.TryParse(args[i + 3], out seed) &&
            int.TryParse(args[i + 4], out port)) {
            socketMode = true;
        }
    }
}

if (socketMode) {
    SocketApplication.Initialize(numParallelGames, useSeparateThreads, seed, port);
    SocketApplication.Start();
} else {
    Application.Initialize();
    Application.Start();
}

