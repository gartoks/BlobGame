using System.Diagnostics;
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
    /// <summary>
    /// The number of parallel games to run.
    /// </summary>
    private static int NumParallelGames { get; set; }
    /// <summary>
    /// Wether or not to run each game on a separate thread or interleave them.
    /// </summary>
    private static bool UseSeparateThreads { get; set; }

    /// <summary>
    /// The port used to connect.
    /// </summary>
    private static int Port { get; set; }

    private static IReadOnlyList<Thread>? Threads { get; set; }

    private static IReadOnlyList<(ClassicGameMode simulation, SocketController controller)>? Games { get; set; }

    internal static void Initialize(int numParallelGames, bool useSeparateThreads, int seed, int port) {
        Seed = seed;
        NumParallelGames = numParallelGames;
        UseSeparateThreads = useSeparateThreads;
        Port = port;

        if (UseSeparateThreads)
            InitializeWithThreads();
        else
            InitializeWithoutThreads();
    }

    private static void InitializeWithThreads() {
        Random random = new Random(Seed);
        Threads = Enumerable.Range(0, NumParallelGames)
            .Select(i => new Thread(() => RunGameThread(i, random.Next())))
            .ToList();
    }

    private static void InitializeWithoutThreads() {
        Random random = new Random(Seed);
        Games = Enumerable.Range(0, NumParallelGames)
            .Select(i => (new ClassicGameMode(random.Next()), new SocketController(i, Port)))
            .ToList();
    }

    internal static void Start() {
        if (UseSeparateThreads)
            StartWithThreads();
        else
            StartWithoutThreads();
    }

    private static void StartWithThreads() {
        foreach (Thread thread in Threads!)
            thread.Start();

        foreach (Thread thread in Threads)
            thread.Join();
    }

    private static void StartWithoutThreads() {
        const float dT = 1f / 60f;

        List<int> runningGames = Enumerable.Range(0, NumParallelGames).ToList();

        foreach ((IGameMode simulation, SocketController controller) in Games!){
            simulation.Load();
            controller.Load();
        }

        while (runningGames.Count > 0) {
            for (int i = 0; i < Games!.Count; i++) {
                if (!runningGames.Contains(i))
                    continue;

                (ClassicGameMode simulation, SocketController controller) = Games[i];

                simulation.Update(dT);
                controller.Update(dT, simulation);

                if (simulation.CanSpawnBlob && controller.SpawnBlob(simulation, out float t)) {
                    t = Math.Clamp(t, 0, 1);
                    simulation.TrySpawnBlob(t);
                }

                if (simulation.IsGameOver || !controller.IsConnected) {
                    runningGames.Remove(i);
                    // send the game over state
                    controller.Update(dT, simulation);
                    Console.WriteLine($"Game {i} ended with score {simulation.Score}. {runningGames.Count} games running.");
                }
            }
        }
    }

    private static void RunGameThread(int gameIndex, int seed) {
        const float dT = 1f / 60f;

        ClassicGameMode simulation = new ClassicGameMode(seed);
        SocketController controller = new SocketController(gameIndex, Port);

        Stopwatch sw = new Stopwatch();
        sw.Start();
        int frameCounter = 1;

        simulation.Load();
        controller.Load();

        while (!simulation.IsGameOver && controller.IsConnected) {
            simulation.Update(dT);
            controller.Update(dT, simulation);

            if (simulation.CanSpawnBlob && controller.SpawnBlob(simulation, out float t)) {
                t = Math.Clamp(t, 0, 1);
                simulation.TrySpawnBlob(t);
            }
            frameCounter++;
        }
        // send the game over state
        controller.Update(dT, simulation);

        Console.WriteLine($"Game {gameIndex} has finished with {simulation.Score} points");
    }
}
