namespace BlobGame.Game.GameControllers;
/// <summary>
/// Represents an interface to control the game.
/// </summary>
internal interface IGameController {
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
    bool SpawnBlob(ISimulation simulation, out float t);

    /// <summary>
    /// Closes any connections and disposes resources needed by the controller
    /// </summary>
    void Close();
}
