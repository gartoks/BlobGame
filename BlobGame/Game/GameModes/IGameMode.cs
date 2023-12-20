using BlobGame.Game.Blobs;
using BlobGame.Game.Util;
using System.Numerics;

namespace BlobGame.Game.GameModes;

internal delegate void BlobEventHandler(IGameMode sender, Vector2 position, int type);
internal delegate void GameEventHandler(IGameMode sender);

/// <summary>
/// Interface which exposes the game's simulation logic to allow outside control and getting the state of the game.
/// </summary>
internal interface IGameMode {
    /// <summary>
    /// Constants derived from the original game. Manually figured out and hard coded.
    /// </summary>
    internal const float GRAVITY = 111.3f;
    internal const float ARENA_WIDTH = 670;
    internal const float ARENA_HEIGHT = 846;
    internal const float ARENA_HEIGHT_LOWER = 750;
    internal const float ARENA_WALL_THICKNESS = 300; // visually: 20
    internal const float ARENA_SPAWN_Y_OFFSET = -22.5f;

    /// <summary>
    /// The id of the game mode. This must never change for a given game mode.
    /// </summary>
    Guid Id { get; }

    /// <summary>
    /// Tha available blob types.
    /// </summary>
    IReadOnlyDictionary<int, BlobData> Blobs { get; }

    /// <summary>
    /// The game's game objects. This includes blobs and walls.
    /// </summary>
    IReadOnlyGameObjectsCollection GameObjects { get; }
    /// <summary>
    /// The type of the currently spawned blob.
    /// </summary>
    int CurrentBlob { get; }
    /// <summary>
    /// The type of the next blob to be spawned.
    /// </summary>
    int NextBlob { get; }
    /// <summary>
    /// The type of the currently held blob.
    /// </summary>
    int HeldBlob { get; }
    /// <summary>
    /// Wether or not the player can currently spawn a blob. This is false when the last spawned blob is still falling.
    /// </summary>
    bool CanSpawnBlob { get; }
    /// <summary>
    /// The current score of the game
    /// </summary>
    int Score { get; }
    /// <summary>
    /// Flag indicating wether or not the game is over. The game is over when a blob "overfills" the arena.
    /// </summary>
    bool IsGameOver { get; }

    /// <summary>
    /// The rotation of the next blob to be spawned.
    /// </summary>
    float SpawnRotation { get; }

    /// <summary>
    /// Event that is fired when a blob is spawned.
    /// </summary>
    event BlobEventHandler OnBlobSpawned;

    /// <summary>
    /// Event that is fired when a newly spawned blob collides for the first time.
    /// </summary>
    event BlobEventHandler OnBlobPlaced;

    /// <summary>
    /// Event that is fired when two blobs combine. The argument is the type of the blob that was created.
    /// </summary>
    event BlobEventHandler OnBlobsCombined;

    /// <summary>
    /// Event that is fired when a blob is destroyed. The argument is the type of the blob that was destroyed.
    /// </summary>
    event BlobEventHandler OnBlobDestroyed;

    /// <summary>
    /// Event that is fired when the game is over.
    /// </summary>
    event GameEventHandler OnGameOver;

    /// <summary>
    /// Loads the game mode. This is called when the game is started. Should be used to initialize the game mode and load resources.
    /// </summary>
    void Load();

    /// <summary>
    /// Attempts to spawn a blob. Returns true if a blob was spawned, false otherwise.
    /// </summary>
    /// <param name="t">The t value where to spawn the blob above the arena. 0 is all the way left, 1 all the way right.</param>
    /// <returns>Returns true if the blob should be spawned; false otherwise.</returns>
    bool TrySpawnBlob(float t);

    /// <summary>
    /// Attempts to hold the current blob. If a blob is already held, the current blob is swapped with the held blob.
    /// </summary>
    void HoldBlob();

    /// <summary>
    /// Used to update the game simulation. Is called every frame. Simulates the physics, handles game object adding and removing and check for game over conditions.
    /// </summary>
    /// <param name="dT"></param>
    void Update(float dT);

    /// <summary>
    /// All available game modes with their keys.
    /// </summary>
    public static IReadOnlyDictionary<string, Type> GameModeTypes { get; } = new Dictionary<string, Type>() {
        { "Toasted", typeof(ToastedGameMode) },
        { "Classic", typeof(ClassicGameMode) },
    };
    /// <summary>
    /// All available game modes with their names.
    /// </summary>
    public static IReadOnlyDictionary<Type, string> GameModeNames { get; } = new Dictionary<Type, string>() {
        { typeof(ToastedGameMode), "Toasted" },
        { typeof(ClassicGameMode), "Classic" },
    };

    /// <summary>
    /// Creates a new game mode instance of the provided type with the provided seed.
    /// </summary>
    /// <param name="gameModeType"></param>
    /// <param name="seed"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException">If the given type is invalid or is missing a proper constructor.</exception>
    public static IGameMode CreateGameMode(Type gameModeType, int seed) {
        if (!typeof(IGameMode).IsAssignableFrom(gameModeType))
            throw new ArgumentException("Game mode type must implement IGameMode", nameof(gameModeType));

        if (!gameModeType.IsClass)
            throw new ArgumentException("Game mode type must be a class", nameof(gameModeType));

        if (gameModeType.IsAbstract)
            throw new ArgumentException("Game mode type must not be abstract", nameof(gameModeType));

        if (!gameModeType.GetConstructors().Any(c => c.GetParameters().Length == 1 && c.GetParameters()[0].ParameterType == typeof(int)))
            throw new ArgumentException("Game mode type must have a constructor with one int parameter", nameof(gameModeType));

        return (IGameMode)Activator.CreateInstance(gameModeType, new object[] { seed })!;
    }
}
