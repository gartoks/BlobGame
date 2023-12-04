using BlobGame.App;
using BlobGame.ResourceHandling;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Rendering;
using SimpleGL.Graphics.Textures;

namespace BlobGame.Game.Tutorial;
internal abstract class TutorialStage {
    protected const int BASE_ZINDEX = 10;

    protected Texture AvatarTexture { get; set; }
    protected Texture LMBTexture { get; set; }
    protected Texture PointerTexture { get; set; }

    internal virtual bool IsFadeInFinished => true;
    internal virtual bool IsFadeOutFinished => true;

    internal virtual void Load() {
        AvatarTexture = ResourceManager.TextureLoader.GetResource("melba_avatar");
        LMBTexture = ResourceManager.TextureLoader.GetResource("lmb");
        PointerTexture = ResourceManager.TextureLoader.GetResource("pointer");
    }

    internal virtual void Unload() {

    }

    internal virtual void DrawFadeIn() {
    }

    internal abstract void Draw();

    internal virtual void DrawFadeOut() {
    }

    protected void DrawLMBHint(float x) {
        Primitives.DrawSprite(new Vector2(x, GameApplication.PROJECTION_HEIGHT - 50), new Vector2(100, 100), new Vector2(0, 1), 0, BASE_ZINDEX + 1, LMBTexture, Color4.White);

        MeshFont font = Fonts.GetGuiFont(80);
        Primitives.DrawText(font, "Hold LMB to continue!", ResourceManager.ColorLoader.GetResource("dark_accent"), new Vector2(x + 125, GameApplication.PROJECTION_HEIGHT - 150), new Vector2(0.5f, 0.5f), 0, BASE_ZINDEX + 1);
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