using BlobGame.App;
using BlobGame.Audio;
using BlobGame.Rendering;
using BlobGame.ResourceHandling;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Rendering;
using SimpleGL.Graphics.Textures;
using SimpleGL.Util.Math;

namespace BlobGame.Game.Tutorial.Stages;
internal class TutorialStage5 : TutorialStage {
    private const float AVATAR_X = 1000;

    private Texture SpeechbubbleTexture { get; set; }

    private AnimatedTexture AnimatedSpeechbubble { get; set; }
    private AnimatedTexture AnimatedAvatarFadeIn { get; set; }
    private AnimatedTexture AnimatedPointer { get; set; }
    private Vector2 PointerAnimationDirection { get; }

    internal override bool IsFadeInFinished => AnimatedAvatarFadeIn.IsFinished;
    internal override bool IsFadeOutFinished => true;

    private bool PlayedSound { get; set; }

    public TutorialStage5() {
        PlayedSound = false;

        PointerAnimationDirection = new Vector2(MathF.Cos(MathF.PI * 3f / 2f), MathF.Sin(MathF.PI * 3f / 2));
    }

    internal override void Load() {
        base.Load();
        SpeechbubbleTexture = ResourceManager.TextureLoader.GetResource("speechbubble");

        AnimatedSpeechbubble = new AnimatedTexture(
            SpeechbubbleTexture,
            2,
            new Vector2(GameApplication.PROJECTION_WIDTH / 2 + 20, GameApplication.PROJECTION_HEIGHT / 2 + 70),
            new Vector2(300, 300),  // TODO find size
            0,
            BASE_ZINDEX,
            new Vector2(0.5f, 0.5f)) {
            ScaleAnimator = t => Vector2.One + new Vector2(0.05f, 0.05f) * GetSpeechbubbleScaleT(t),
            RotationAnimator = t => MathF.PI / 128 * GetSpeechbubbleRotationT(t)
        };

        AnimatedAvatarFadeIn = new AnimatedTexture(
            AvatarTexture,
            0.65f,
            new Vector2(AVATAR_X, GameApplication.PROJECTION_HEIGHT + AvatarTexture.Height),
            new Vector2(944, 1432),
            0,
            BASE_ZINDEX,
            new Vector2(0, 1)) {
            PositionAnimator = t => new Vector2(0, -GetAvatarPositionT(t) * AvatarTexture.Height / 2f)
        };

        AnimatedPointer = new AnimatedTexture(
            PointerTexture,
            0.5f,
            new Vector2(0.193f * GameApplication.PROJECTION_WIDTH, 0.3f * GameApplication.PROJECTION_HEIGHT),
            new Vector2(256, 256),
            -55f.ToRad(),
            BASE_ZINDEX,
            Vector2.One / 2f) {
            PositionAnimator = t => PointerAnimationDirection * 10 * -MathF.Sin(MathF.Tau * t)
        };
    }

    internal override void Unload() {
        base.Unload();
    }

    internal override void DrawFadeIn() {
        if (AnimatedAvatarFadeIn.IsReady)
            AnimatedAvatarFadeIn.Start();
        AnimatedAvatarFadeIn.Render();
    }

    internal override void Draw() {
        if (!PlayedSound) {
            AudioManager.PlaySound("tutorial_5");
            PlayedSound = true;
        }

        Primitives.DrawSprite(new Vector2(AVATAR_X, GameApplication.PROJECTION_HEIGHT - AvatarTexture.Height / 2), new Vector2(944, 1432), new Vector2(0.5f, 0.5f), 0, BASE_ZINDEX, AvatarTexture, Color4.White);

        if (AnimatedPointer.IsReady)
            AnimatedPointer.Start();

        AnimatedPointer.Render();

        if (AnimatedPointer.IsFinished)
            AnimatedPointer.Start();

        if (AnimatedAvatarFadeIn.IsFinished) {
            DrawSpeechBubble();

            MeshFont font = Fonts.GetGuiFont(50);
            Primitives.DrawText(font, "Here is your current\nhighscore!", ResourceManager.ColorLoader.GetResource("dark_accent"), new Vector2(600, GameApplication.PROJECTION_HEIGHT / 2 - 100), new Vector2(0.5f, 0.5f), 0, BASE_ZINDEX + 1);

            DrawLMBHint(50);
        }
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

    private float GetAvatarPositionT(float t) {
        float tmp = 1.3f * t - 1;
        return 1.1f * (-(tmp * tmp) + 1);
    }

    private float GetSpeechbubbleScaleT(float t) {
        return 0.5f * (MathF.Sin(MathF.Tau * t - MathF.PI / 2) + 1);
    }

    private float GetSpeechbubbleRotationT(float t) {
        return -MathF.Sin(MathF.Tau * t);
    }

}
