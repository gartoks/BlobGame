using BlobGame.ResourceHandling;
using OpenTK.Mathematics;
using SimpleGL;
using SimpleGL.Graphics;
using SimpleGL.Graphics.GLHandling;
using SimpleGL.Graphics.Rendering;

namespace BlobGame.App;
internal sealed partial class GameApplication : Application {

    public static float RenderGameTime { get; private set; }
    private Renderer GameRenderer { get; }
    private Renderer GuiRenderer { get; }

    public override void OnRenderStart() {
        Window.Title = NAME;
        Window.ClientSize = new Vector2i(PROJECTION_WIDTH, PROJECTION_HEIGHT);
    }

    public override void OnRender(float deltaTime) {
        GLHandler.ClearColor = ResourceManager.GetResourceState("background") != eResourceLoadStatus.Loaded ? Color4.Black : ResourceManager.ColorLoader.GetResource("background");

        RenderGameTime += deltaTime;

        GameRenderer.BeginRendering(Camera.ProjectionMatrix);

        Game.GameManager.Render(deltaTime);

        if (GameApplication.DRAW_DEBUG) {
            int fps = Window.Fps;
            MeshFont font = Fonts.GetFont("Consolas", 16);
            Primitives.DrawText(font, fps.ToString(), Color4.Lime, new Vector2(10, 10), Vector2.Zero, 0, int.MaxValue);

            Vector2 mPos = Input.GetMousePosition();
            Primitives.DrawText(font, $"{mPos.X:0.}, {mPos.Y:0.}", Color4.Magenta, new Vector2(30, 30), Vector2.Zero, 0, int.MaxValue);
        }

        GameRenderer.EndRendering();

        GuiRenderer.BeginRendering(Camera.ProjectionMatrix);
        Game.GameManager.RenderGui(deltaTime);
        GuiRenderer.EndRendering();
    }

    public override void OnRenderStop() {
    }

}
