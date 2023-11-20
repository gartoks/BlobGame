using BlobGame.Drawing;
using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using System.Numerics;

namespace BlobGame.Game.Tutorial;
internal abstract class TutorialStage {
    protected TextureResource AvatarTexture { get; set; }
    protected TextureResource LMBTexture { get; set; }
    protected TextureResource PointerTexture { get; set; }

    internal virtual bool IsFadeInFinished => true;
    internal virtual bool IsFadeOutFinished => true;

    internal virtual void Load() {
        AvatarTexture = ResourceManager.TextureLoader.Get("melba_avatar");
        LMBTexture = ResourceManager.TextureLoader.Get("lmb");
        PointerTexture = ResourceManager.TextureLoader.Get("pointer");
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

        Renderer.GuiFont.Draw(
            "Hold LMB to continue!",
            80,
            ResourceManager.ColorLoader.Get("dark_accent"),
            new Vector2(x + 125, Application.BASE_HEIGHT - 150));
    }

    //protected void DrawArrow(Vector2 arrowPos, float angle, Vector2 start, Vector2 end, Vector2 startControl, Vector2 encControl) {
    //    Vector2 size = new Vector2(Application.BASE_WIDTH, Application.BASE_HEIGHT);
    //    Vector2 arrowCenter = new Vector2(0.4131147540983606f, 0.6012658227848f);

    //    ArrowHeadTexture.Draw(
    //        arrowPos * size,
    //        arrowCenter,
    //        new Vector2(0.3f, 0.3f),
    //        angle);

    //    Raylib.DrawLineBezierCubic(
    //        start * size,
    //        end * size,
    //        startControl * size,
    //        encControl * size,
    //        60,
    //        ResourceManager.ColorLoader.Get("dark_accent").Resource);
    //    Raylib.DrawLineBezierCubic(
    //        start * size,
    //        end * size,
    //        startControl * size,
    //        encControl * size,
    //        40,
    //        ResourceManager.ColorLoader.Get("highlight").Resource);
    //}

}