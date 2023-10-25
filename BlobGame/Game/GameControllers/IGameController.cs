using BlobGame.Game.Scenes;

namespace BlobGame.Game.GameControllers;
internal interface IGameController {
    float GetCurrentT(GameScene scene);
    bool SpawnFruit(GameScene scene, out float t);
}
