using BlobGame.App;
using BlobGame.Game.Scenes;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.GameControllers;
internal class MouseController : IGameController {

    public float GetCurrentT(GameScene scene) {
        Vector2 mPos = scene.ScreenToArenaPosition(Raylib.GetMousePosition());
        float t = mPos.X / Simulation.ARENA_WIDTH;
        return t;
    }

    public bool SpawnFruit(GameScene scene, out float t) {
        t = -1;

        if (!scene.GameSim.CanSpawnBlob)
            return false;

        if (!InputHandler.IsMouseButtonActive(MouseButton.MOUSE_BUTTON_LEFT))
            return false;


        t = GetCurrentT(scene);
        return true;
    }

}
