using BlobGame.Game.Blobs;
using BlobGame.Game.Util;

namespace BlobGame.Game.GameModes;

/// <summary>
/// Interface which exposes the game's simulation logic to allow outside control and getting the state of the game.
/// </summary>
internal interface IGameMode {
    /// <summary>
    /// The game's game objects. This includes blobs and walls.
    /// </summary>
    IReadOnlyGameObjectsCollection GameObjects { get; }
    /// <summary>
    /// The type of the currently spawned blob.
    /// </summary>
    eBlobType CurrentBlob { get; }
    /// <summary>
    /// The type of the next blob to be spawned.
    /// </summary>
    eBlobType NextBlob { get; }
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
    /// Event that is fired when a blob is spawned.
    /// </summary>
    event Action<eBlobType> OnBlobSpawned;

    /// <summary>
    /// Event that is fired when a newly spawned blob collides for the first time.
    /// </summary>
    event Action<eBlobType> OnBlobPlaced;

    /// <summary>
    /// Event that is fired when two blobs combine. The argument is the type of the blob that was created.
    /// </summary>
    event Action<eBlobType> OnBlobsCombined;

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
    /// Used to update the game simulation. Is called every frame. Simulates the physics, handles game object adding and removing and check for game over conditions.
    /// </summary>
    /// <param name="dT"></param>
    void Update(float dT);

    /// <summary>
    /// All available game modes with their keys.
    /// </summary>
    public static IReadOnlyDictionary<string, Type> GameModeTypes { get; } = new Dictionary<string, Type>() {
        { "Classic", typeof(ClassicGameMode) },
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
