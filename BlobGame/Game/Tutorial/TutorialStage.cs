using BlobGame.Drawing;
using BlobGame.ResourceHandling;
using System.Numerics;

namespace BlobGame.Game.Tutorial;
internal abstract class TutorialStage {
    protected TextureResource AvatarTexture { get; set; }
    protected TextureResource LMBTexture { get; set; }

    internal virtual bool IsFadeInFinished => true;
    internal virtual bool IsFadeOutFinished => true;

    internal virtual void Load() {
        AvatarTexture = ResourceManager.GetTexture("melba_avatar");
        LMBTexture = ResourceManager.GetTexture("lmb");
    }

    internal virtual void Unload() {

    }

    internal virtual void DrawFadeIn() {
    }

    internal abstract void Draw();

    internal virtual void DrawFadeOut() {
    }

    protected void DrawLMBHint(float x) {
        LMBTexture.Draw(
            new Vector2(x, Application.BASE_HEIGHT - 50),
            new Vector2(0, 1),
            new Vector2(0.4f, 0.4f));

        Renderer.Font.Draw(
            "Hold LMB to continue!",
            80,
            ResourceManager.GetColor("dark_accent"),
            new Vector2(x + 125, Application.BASE_HEIGHT - 150));
    }
}