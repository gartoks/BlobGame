using BlobGame.App;
using BlobGame.Audio;
using BlobGame.Rendering;
using BlobGame.ResourceHandling;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Rendering;
using SimpleGL.Graphics.Textures;
using static SimpleGL.Util.Math.MathUtils;

namespace BlobGame.Game.Tutorial.Stages;
internal class TutorialStage6 : TutorialStage {
    private const float AVATAR_X = 1000;

    private Texture SpeechbubbleTexture { get; set; }

    private AnimatedTexture AnimatedSpeechbubble { get; set; }
    private AnimatedTexture AnimatedPointer { get; set; }
    private Vector2 PointerAnimationDirection { get; }

    internal override bool IsFadeInFinished => true;
    internal override bool IsFadeOutFinished => true;

    private bool PlayedSound { get; set; }

    public TutorialStage6() {
        PlayedSound = false;

        PointerAnimationDirection = new Vector2(MathF.Cos(270f.ToRad()), MathF.Sin(270f.ToRad()));
    }

    internal override void Load() {
        base.Load();
        SpeechbubbleTexture = ResourceManager.TextureLoader.GetResource("speechbubble");

        AnimatedSpeechbubble = new AnimatedTexture(
            SpeechbubbleTexture,
            2,
            new Vector2(GameApplication.PROJECTION_WIDTH / 2 + 20, GameApplication.PROJECTION_HEIGHT / 2 + 70),
            Vector2.One,
            0,
            10,
            pivot: new Vector2(0.5f, 0.5f)) {
            ScaleAnimator = t => Vector2.One + new Vector2(0.05f, 0.05f) * GetSpeechbubbleScaleT(t),
            RotationAnimator = t => MathF.PI / 128 * GetSpeechbubbleRotationT(t)
        };

        AnimatedPointer = new AnimatedTexture(
            PointerTexture,
            0.5f,
            new Vector2(0.193f * GameApplication.PROJECTION_WIDTH, 0.43f * GameApplication.PROJECTION_HEIGHT),
            Vector2.One / 2f,
            180f.ToRad(),
            12,
            Vector2.One / 2f) {
            PositionAnimator = t => PointerAnimationDirection * 10 * -MathF.Sin(MathF.Tau * t)
        };
    }

    internal override void Unload() {
        base.Unload();
    }

    internal override void DrawFadeIn() {
    }

    internal override void Draw() {
        if (!PlayedSound) {
            AudioManager.PlaySound("tutorial_6");
            PlayedSound = true;
        }


        Primitives.DrawSprite(new Vector2(AVATAR_X, GameApplication.PROJECTION_HEIGHT - AvatarTexture.Height / 2), new Vector2(200, 200), new Vector2(0.5f, 0.5f), 0, 9, AvatarTexture, Color4.White);

        if (AnimatedPointer.IsReady)
            AnimatedPointer.Start();

        AnimatedPointer.Render();

        if (AnimatedPointer.IsFinished)
            AnimatedPointer.Start();

        DrawSpeechBubble();

        MeshFont font = Fonts.GetGuiFont(50);
        Primitives.DrawText(font, "These are your highscores\nfrom today!", ResourceManager.ColorLoader.GetResource("dark_accent"), new Vector2(600, GameApplication.PROJECTION_HEIGHT / 2 - 100), new Vector2(0.5f, 0.5f), 0, 11);

        DrawLMBHint(50);
    }

    internal override void DrawFadeOut() {
    }

    private void DrawSpeechBubble() {
        if (AnimatedSpeechbubble.IsReady)
            AnimatedSpeechbubble.Start();

        AnimatedSpeechbubble.Render();

        if (AnimatedSpeechbubble.IsFinished)
            AnimatedSpeechbubble.Start();
    }

    private float GetSpeechbubbleScaleT(float t) {
        return 0.5f * (MathF.Sin(MathF.Tau * t - MathF.PI / 2) + 1);
    }

    private float GetSpeechbubbleRotationT(float t) {
        return -MathF.Sin(MathF.Tau * t);
    }

}
