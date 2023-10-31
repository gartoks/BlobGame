using BlobGame.Game.GameModes;

namespace BlobGame.Game.GameControllers;
/// <summary>
/// Represents an interface to control the game.
/// </summary>
internal interface IGameController {
    /// <summary>
    /// Loads any resources needed by the controller or registeres potential hotkeys.
    /// </summary>
    void Load();

    /// <summary>
    /// Retrieves the current value of t, which represents the position of the dropper above the arena.
    /// </summary>
    /// <returns>The current value of t.</returns>
    float GetCurrentT();

    /// <summary>
    /// Attempts to spawn a blob in the provided game simulation.
    /// </summary>
    /// <param name="simulation">The game simulation in which to spawn the blob.</param>
    /// <param name="t">The t value at which the blob is spawned, which represents the position of the dropper above the arena..</param>
    /// <returns>True if blob spawning was attempted, otherwise false.</returns>
    bool SpawnBlob(IGameMode simulation, out float t);

    /// <summary>
    /// Runs every frame.
    /// </summary>
    void Update(float dT, IGameMode simulation);

    /// <summary>
    /// Closes any connections and disposes resources needed by the controller or unregisters potential hotkeys.
    /// </summary>
    void Close();

    /// <summary>
    /// All available game controllers with their keys.
    /// </summary>
    public static IReadOnlyDictionary<string, Type> ControllerTypes { get; } = new Dictionary<string, Type>() {
        { "Mouse", typeof(MouseController) },
        { "Keyboard", typeof(KeyboardController) },
        //{ "Socket", typeof(SocketController) },   // TODO
    };

    /// <summary>
    /// Creates a new game controlelr instance of the provided type.
    /// </summary>
    /// <param name="controllerType"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException">If the given type is invalid or is missing a proper constructor.</exception>
    public static IGameController CreateGameController(Type controllerType) {
        if (!typeof(IGameController).IsAssignableFrom(controllerType))
            throw new ArgumentException("Game mode type must implement IGameMode", nameof(controllerType));

        if (!controllerType.IsClass)
            throw new ArgumentException("Game controller type must be a class", nameof(controllerType));

        if (controllerType.IsAbstract)
            throw new ArgumentException("Game controller type must not be abstract", nameof(controllerType));

        if (!controllerType.GetConstructors().Any(c => c.GetParameters().Length == 0))
            throw new ArgumentException("Game controller type must have a constructor with no parameters", nameof(controllerType));

        return (IGameController)Activator.CreateInstance(controllerType)!;
    }
}
