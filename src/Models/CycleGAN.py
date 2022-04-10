import tensorflow as tf
from Models.PatchGAN import PatchGANGenerator, PatchGANDiscriminator

from Configs.ConfigCycleGAN import ConfigCycleGAN

class CycleGAN(tf.keras.Model):

    def __init__(self, config: ConfigCycleGAN):
        super().__init__()

        self.config = config

        self.adverserial_loss = tf.keras.losses.MeanSquaredError()
        self.cycle_loss = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss = tf.keras.losses.MeanAbsoluteError()

        self.generator_AtoB_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["hyperparameters"]["learning_rate_generator_AtoB"], beta_1=self.config["hyperparameters"]["beta_1_generator_AtoB"])
        self.generator_BtoA_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["hyperparameters"]["learning_rate_generator_BtoA"], beta_1=self.config["hyperparameters"]["beta_1_generator_BtoA"])

        self.discriminator_A_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["hyperparameters"]["learning_rate_discriminator_A"], beta_1=self.config["hyperparameters"]["beta_1_discriminator_A"])
        self.discriminator_B_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["hyperparameters"]["learning_rate_discriminator_B"], beta_1=self.config["hyperparameters"]["beta_1_discriminator_B"])

        self.cycle_weight = self.config["hyperparameters"]["cycle_weight"]
        self.identity_weight = self.config["hyperparameters"]["identity_weight"]

        self.generator_AtoB: tf.keras.Model = PatchGANGenerator()
        self.generator_BtoA: tf.keras.Model = PatchGANGenerator()

        self.discriminator_A: tf.keras.Model = PatchGANDiscriminator()
        self.discriminator_B: tf.keras.Model = PatchGANDiscriminator()

    def train_step(self, datasets):
        real_data_A, real_data_B = datasets
        with tf.GradientTape(persistent=True) as tape:
            # generates image with type B from image type A
            fake_data_B = self.generator_AtoB(real_data_A, training=True)
            # generates image with type A from image type B
            fake_data_A = self.generator_BtoA(real_data_B, training=True)

            # image type A is translated to type B and back to A
            back_cycled_A = self.generator_BtoA(fake_data_B, training=True)
            back_cycled_B = self.generator_AtoB(fake_data_A, training=True)

            # identity mapping: generates image type X from image type X
            identity_B = self.generator_AtoB(real_data_B, training=True)
            identity_A = self.generator_BtoA(real_data_A, training=True)

            #train discriminator with real data and later fake data
            discriminator_real_A = self.discriminator_A(real_data_A, training=True)
            discriminator_fake_A = self.discriminator_A(fake_data_A, training=True)

            # train discriminator with real data and later fake data
            discriminator_real_B = self.discriminator_B(real_data_B, training=True)
            discriminator_fake_B = self.discriminator_B(fake_data_B, training=True)

            # calculates generator loss
            generator_AtoB_loss = self.generator_fake_adverserial_loss(discriminator_fake_B)
            generator_BtoA_loss = self.generator_fake_adverserial_loss(discriminator_fake_A)

            # calculates back cycle data
            back_cycle_loss_A = self.cycle_loss(real_data_A, back_cycled_A) * self.cycle_weight
            back_cycle_loss_B = self.cycle_loss(real_data_B, back_cycled_B) * self.cycle_weight

            # Calculates identity loss
            identity_loss_A = self.identity_loss(real_data_A, identity_A) * self.cycle_weight * self.identity_weight
            identity_loss_B = self.identity_loss(real_data_B, identity_B) * self.cycle_weight * self.identity_weight

            # Sum Losses up
            generator_AtoB_total_loss = generator_AtoB_loss + back_cycle_loss_B + identity_loss_B
            generator_BtoA_total_loss = generator_BtoA_loss + back_cycle_loss_A + identity_loss_A

            # Calculate Discriminator losses
            discriminator_A_loss = self.discriminator_loss(discriminator_real_A, discriminator_fake_A)
            discriminator_B_loss = self.discriminator_loss(discriminator_real_B, discriminator_fake_B)

        # Get Gradients via Losses for Generators
        gradients_generator_AtoB = tape.gradient(generator_AtoB_total_loss, self.generator_AtoB.trainable_variables)
        gradients_generator_BtoA = tape.gradient(generator_BtoA_total_loss, self.generator_BtoA.trainable_variables)

        # Get Gradients via Losses for Discriminators
        gradients_discriminator_A = tape.gradient(discriminator_A_loss, self.discriminator_A.trainable_variables)
        gradients_discriminator_B = tape.gradient(discriminator_B_loss, self.discriminator_B.trainable_variables)

        # Apply Gradients for Generators
        self.generator_AtoB_optimizer.apply_gradients(zip(gradients_generator_AtoB, self.generator_AtoB.trainable_variables))
        self.generator_BtoA_optimizer.apply_gradients(zip(gradients_generator_BtoA, self.generator_BtoA.trainable_variables))

        # Apply Gradients for Discriminators
        self.discriminator_A_optimizer.apply_gradients(zip(gradients_discriminator_A, self.discriminator_A.trainable_variables))
        self.discriminator_B_optimizer.apply_gradients(zip(gradients_discriminator_B, self.discriminator_B.trainable_variables))

        return {
            "Generator_AtoB": generator_AtoB_total_loss,
            "Generator_BtoA": generator_BtoA_total_loss,
            "Discriminator_A_Loss": discriminator_A_loss,
            "Discriminator_B_Loss": discriminator_B_loss,
        }

    def generator_fake_adverserial_loss(self, discriminator_fake_output):
        fake_loss = self.adverserial_loss(tf.ones_like(discriminator_fake_output), discriminator_fake_output)
        return fake_loss

    def discriminator_loss(self, real, fake):
        real_loss = self.adverserial_loss(tf.ones_like(real), real)
        fake_loss = self.adverserial_loss(tf.zeros_like(fake), fake)
        return (real_loss + fake_loss) * 0.5

