using BlobGame.App;
using BlobGame.Game.Scenes;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.GameControllers;

internal class MouseController : IGameController {
    /// <summary>
    /// Represents the game scene associated with this controller.
    /// </summary>
    private GameScene Scene { get; }

    public MouseController(GameScene scene) {
        Scene = scene;
    }

    /// <summary>
    /// Retrieves the current value of t, which represents the position of the dropper above the arena.
    /// </summary>
    /// <returns>The current value of t.</returns>
    public float GetCurrentT() {
        Vector2 mPos = Scene.ScreenToArenaPosition(Raylib.GetMousePosition());
        float t = mPos.X / Simulation.ARENA_WIDTH;
        return t;
    }

    /// <summary>
    /// Attempts to spawn a blob in the provided game simulation.
    /// </summary>
    /// <param name="simulation">The game simulation in which to spawn the blob.</param>
    /// <param name="t">The t value at which the blob is spawned, which represents the position of the dropper above the arena..</param>
    /// <returns>True if blob spawning was attempted, otherwise false.</returns>
    public bool SpawnBlob(ISimulation simulation, out float t) {
        t = -1;

        if (!simulation.CanSpawnBlob)
            return false;

        if (!Input.IsMouseButtonActive(MouseButton.MOUSE_BUTTON_LEFT))
            return false;

        Input.WasMouseHandled[MouseButton.MOUSE_BUTTON_LEFT] = true;

        t = GetCurrentT();
        return true;
    }
    public void Update(ISimulation simulation) { }

    public void Close() { }

}
