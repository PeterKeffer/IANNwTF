import gc
import logging

import tensorflow as tf
from src.Models.PatchGAN import PatchGANGenerator, PatchGANDiscriminator

class CycleGAN(tf.keras.Model):

    def __init__(self, generator_AtoB, generator_BtoA, discriminator_A, discriminator_B):
        super(CycleGAN, self).__init__()

        self.cycle_weight = 10.0
        self.identity_weight = 0.5

        self.generator_AtoB: tf.keras.Model = generator_AtoB
        self.generator_BtoA: tf.keras.Model = generator_BtoA

        self.discriminator_A: tf.keras.Model = discriminator_A
        self.discriminator_B: tf.keras.Model = discriminator_B


    def compile(
        self,
        generator_AtoB_optimizer,
        generator_BtoA_optimizer,
        discriminator_A_optimizer,
        discriminator_B_optimizer,
        adverserial_loss,
        cycle_loss,
        identity_loss,
        generator_fake_adverserial_loss,
        discriminator_loss,
    ):
        super(CycleGAN, self).compile()
        self.generator_AtoB_optimizer = generator_AtoB_optimizer
        self.generator_BtoA_optimizer = generator_BtoA_optimizer
        self.discriminator_A_optimizer = discriminator_A_optimizer
        self.discriminator_B_optimizer = discriminator_B_optimizer
        self.adverserial_loss = adverserial_loss
        self.cycle_loss = cycle_loss
        self.identity_loss = identity_loss
        self.generator_fake_adverserial_loss = generator_fake_adverserial_loss
        self.discriminator_loss = discriminator_loss

    def train_step(self, datasets):
        real_data_A, real_data_B = datasets
        # TODO persistent auf False setzen um Overflow zu debuggen
        with tf.GradientTape(persistent=True) as tape:
            # generates image with type B from image type A
            fake_data_B = self.generator_AtoB(real_data_A, training=True)
            # generates image with type A from image type B
            fake_data_A = self.generator_BtoA(real_data_B, training=True)

            # image type A is translated to type B and back to A
            back_cycled_A = self.generator_BtoA(fake_data_B, training=True)
            back_cycled_B = self.generator_AtoB(fake_data_A, training=True)

            # identity mapping: generates image type X from image type X
            identity_A = self.generator_BtoA(real_data_A, training=True)
            identity_B = self.generator_AtoB(real_data_B, training=True)

            #train discriminator with real data and later fake data
            discriminator_real_A = self.discriminator_A(real_data_A, training=True)
            discriminator_fake_A = self.discriminator_A(fake_data_A, training=True)

            # train discriminator with real data and later fake data
            discriminator_real_B = self.discriminator_B(real_data_B, training=True)
            discriminator_fake_B = self.discriminator_B(fake_data_B, training=True)

            # calculates generator loss
            generator_AtoB_loss = self.generator_fake_adverserial_loss(discriminator_fake_B)
            generator_BtoA_loss = self.generator_fake_adverserial_loss(discriminator_fake_A)

            #calculates back cycle data
            back_cycle_loss_AtoB = self.cycle_loss(real_data_A, back_cycled_A) * self.cycle_weight
            back_cycle_loss_BtoA = self.cycle_loss(real_data_B, back_cycled_B) * self.cycle_weight

            identity_loss_A = self.identity_loss(real_data_A, identity_A) * self.cycle_weight * self.identity_weight
            identity_loss_B = self.identity_loss(real_data_B, identity_B) * self.cycle_weight * self.identity_weight

            generator_AtoB_total_loss = generator_AtoB_loss + back_cycle_loss_BtoA + identity_loss_B
            generator_BtoA_total_loss = generator_BtoA_loss + back_cycle_loss_AtoB + identity_loss_A

            discriminator_A_loss = self.discriminator_loss(discriminator_real_A, discriminator_fake_A)
            discriminator_B_loss = self.discriminator_loss(discriminator_real_B, discriminator_fake_B)

        gradients_generator_AtoB = tape.gradient(generator_AtoB_total_loss, self.generator_AtoB.trainable_variables)
        gradients_generator_BtoA = tape.gradient(generator_BtoA_total_loss, self.generator_BtoA.trainable_variables)

        gradients_discriminator_A = tape.gradient(discriminator_A_loss, self.discriminator_A.trainable_variables)
        gradients_discriminator_B = tape.gradient(discriminator_B_loss, self.discriminator_B.trainable_variables)

        self.generator_AtoB_optimizer.apply_gradients(zip(gradients_generator_AtoB, self.generator_AtoB.trainable_variables))
        self.generator_BtoA_optimizer.apply_gradients(zip(gradients_generator_BtoA, self.generator_BtoA.trainable_variables))
        
        self.discriminator_A_optimizer.apply_gradients(zip(gradients_discriminator_A, self.discriminator_A.trainable_variables))
        self.discriminator_B_optimizer.apply_gradients(zip(gradients_discriminator_B, self.discriminator_B.trainable_variables))

        return {
            "Generator_AtoB": generator_AtoB_total_loss,
            "Generator_BtoA": generator_BtoA_total_loss,
            "Discriminator_A_Loss": discriminator_A_loss,
            "Discriminator_B_Loss": discriminator_B_loss,
        }


