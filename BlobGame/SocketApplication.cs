using BlobGame.Game.GameControllers;
using BlobGame.Game.GameModes;

namespace BlobGame;
/// <summary>
/// Static alternative application to run the game in a server-client architecture.
/// </summary>
internal static class SocketApplication {
    /// <summary>
    /// The base seed for the games.
    /// </summary>
    private static int Seed { get; set; }

    private static Type GameModeType { get; set; }
    /// <summary>
    /// Wether or not to run each game on a separate thread or interleave them.
    /// </summary>
    private static bool UseSeparateThreads { get; set; }

    /// <summary>
    /// The port used to connect.
    /// </summary>
    private static int Port { get; set; }

    internal static void Initialize(bool useSeparateThreads, int seed, int port, string gameModeKey) {
        Seed = seed;
        GameModeType = IGameMode.GameModeTypes[gameModeKey];
        UseSeparateThreads = useSeparateThreads;
        Port = port;
    }

    internal static void Start() {
        if (UseSeparateThreads)
            StartWithThreads();
        else
            StartWithoutThreads();
    }

    private static void StartWithThreads() {
        List<Thread> threads = new();

        Console.WriteLine($"Starting socket mode on port {Port}...");
        SocketController.Load(Port);

        ulong numGames = 0;

        while (true){
            SocketController controller = new((int)numGames);
            controller.Load();
            Thread thread = new(() => RunGameThread((int)numGames, Random.Shared.Next(), controller));
            threads.Add(thread);
            thread.Start();
            numGames++;

            for (int i = threads.Count-1; i>=0; i--){
                if (!thread.IsAlive){
                    thread.Join();
                    threads.RemoveAt(i);
                }
            }

        }

        SocketController.Unload();
    }

    private static void StartWithoutThreads() {
        const float dT = 1f / 60f;

        Console.WriteLine($"Starting socket mode on port {Port}...");
        SocketController.Load(Port);

        List<(IGameMode simulation, SocketController controller, ulong i)> runningGames = new();
        ulong numGames = 0;

        while (true){
            while (SocketController.HasPendingConnections){
                IGameMode simulation = new ClassicGameMode(Random.Shared.Next());
                SocketController controller = new SocketController((int)numGames);

                simulation.Load();
                controller.Load();
                runningGames.Add((simulation, controller, numGames));
                numGames++;
            }

            foreach (var game in runningGames){
                (IGameMode simulation, SocketController controller, ulong i) = game;

                simulation.Update(dT);
                controller.Update(dT, simulation);

                if (simulation.CanSpawnBlob && controller.SpawnBlob(simulation, out float t)) {
                    t = Math.Clamp(t, 0, 1);
                    simulation.TrySpawnBlob(t);
                }
            }


            for (int i = runningGames.Count-1; i>=0; i--){
                (IGameMode simulation, SocketController controller, ulong gameIndex) = runningGames[i];
                if (simulation.IsGameOver || !controller.IsConnected) {
                    runningGames.RemoveAt(i);
                    // send the game over state
                    controller.Update(dT, simulation);
                    Console.WriteLine($"Game {gameIndex} ended with score {simulation.Score}.");
                    controller.Close();
                }
            }
        }

        SocketController.Unload();
    }

    private static void RunGameThread(int gameIndex, int seed, SocketController controller) {
        const float dT = 1f / 60f;

        ClassicGameMode simulation = new ClassicGameMode(seed);
        simulation.Load();

        while (!simulation.IsGameOver && controller.IsConnected) {
            simulation.Update(dT);
            controller.Update(dT, simulation);

            if (simulation.CanSpawnBlob && controller.SpawnBlob(simulation, out float t)) {
                t = Math.Clamp(t, 0, 1);
                simulation.TrySpawnBlob(t);
            }
        }
        // send the game over state
        controller.Update(dT, simulation);

        Console.WriteLine($"Game {gameIndex} has finished with {simulation.Score} points");
        controller.Close();
    }
}
