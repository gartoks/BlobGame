using nkast.Aether.Physics2D.Common;
using nkast.Aether.Physics2D.Dynamics;
using Raylib_CsLo;

namespace BlobGame.Game;
internal sealed class Wall : GameObject {
    protected override Fixture Fixture { get; }

    private float Width { get; }
    private float Height { get; }

    public Wall(string name, World world, Rectangle bounds)
        : base(name, world, new Vector2(bounds.x, bounds.y), 0, BodyType.Static) {
        Width = bounds.width;
        Height = bounds.height;

        Fixture = Body.CreateRectangle(Width / 10f, Height / 10f, 1, new Vector2(Width / 10f / 2, Height / 10f / 2));
        Fixture.Restitution = 0;
        Fixture.Friction = 0.35f;
        ZIndex = 0;
    }

    protected internal override void DrawInternal() {
        if (RaylibApp.DRAW_DEBUG)
            Raylib.DrawRectangleLinesEx(new Rectangle(0, 0, Width, Height), 2, Raylib.WHITE);
    }
}
