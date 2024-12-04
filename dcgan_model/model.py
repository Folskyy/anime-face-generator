# %% [code]
import os
import sys
import json
import time
import warnings

import keras
import numpy as np
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Configura o formato do tensor de imagem (batch_size, height, width, channels)
tf.keras.backend.set_image_data_format('channels_last')
# inicializador para pesos em uma camada da RN utilizando uma distribuição normal
init = tf.keras.initializers.RandomNormal(stddev=0.02)

def build_generator(NOISE_DIM=100):
    """
    Arquitetura do gerador. (4deconv, stride=2)
    Args:
        NOISE_DIM (int): Tamanho do vetor de rúido.
    """
    generator = keras.models.Sequential([
        layers.Dense(4*4*1024, kernel_initializer=init, input_dim=NOISE_DIM),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape(target_shape=(4, 4, 1024)),
        
        layers.Conv2DTranspose(512, 3, strides=2, padding="same", use_bias=False, kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(256, 3, strides=2, padding="same", use_bias=False, kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(128, 3, strides=2, padding="same", use_bias=False, kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(64, 3, strides=2, padding="same", use_bias=False, kernel_initializer=init),
        layers.LeakyReLU(),
        # Função de ativação ideal para valores no intervalo [-1, 1]
        layers.Conv2DTranspose(3, 3, activation="tanh", padding="same")
    ])

    generator.summary()
    return generator

def build_discriminator():
    """
    Arquitetura do discriminador. (4conv, stride=2)
    """
    discriminator = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(64, 64, 3)),

        layers.Conv2D(128, 3, strides=2, padding="same", use_bias=False, kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2D(256, 3, strides=2, padding="same", use_bias=False, kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dropout(.25),

        layers.Conv2D(512, 3, strides=2, padding="same", use_bias=False, kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dropout(.25),
        
        layers.Conv2D(1024, 3, strides=2, padding="same", use_bias=False, kernel_initializer=init),
        layers.LeakyReLU(),

        layers.Flatten(),
        layers.Dense(1, activation="sigmoid")
    ])

    discriminator.summary()
    return discriminator

# tf.config.run_functions_eagerly(True)  # Força execução imediata (sem otimização)
class GAN:
    """
    Classe que implementa uma Rede Generativa Adversarial (GAN), integrando as arquiteturas de
    gerador e discriminador.
    
    Atributos principais:
        generator (tf.keras.Model): Modelo do gerador da GAN.
        discriminator (tf.keras.Model): Modelo do discriminador da GAN.
        generator_optimizer (tf.keras.optimizers.Adam): Otimizador para o gerador.
        discriminator_optimizer (tf.keras.optimizers.Adam): Otimizador para o discriminador.
        history (dict): Histórico de perdas e acurácias do gerador e discriminador.
    """
    def __init__(self, build_generator, build_discriminator, GEN_LR=0.0002, DISC_LR=0.0001,
                 BATCH_SIZE=64, NOISE_DIM=100, EPOCHS=100):
        """
        Args:
            build_generator (function): Função que constrói a arquitetura do gerador.
            build_discriminator (function): Função que constrói a arquitetura do discriminador.
            GEN_LR (float): Taxa de aprendizado do gerador. Default é 0.00002.
            DISC_LR (float): Taxa de aprendizado do discriminador. Default é 0.00001.
            BATCH_SIZE (int): Tamanho do batch de treinamento. Default é 64.
            NOISE_DIM (int): Dimensão do vetor de ruído. Default é 100.
            EPOCHS (int): Número de épocas de treinamento. Default é 100.
        """
        self.BATCH_SIZE = BATCH_SIZE
        self.NOISE_DIM = NOISE_DIM
        self.EPOCHS = EPOCHS

        self.generator = build_generator(self.NOISE_DIM)
        self.discriminator = build_discriminator()

        self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=.1)
        self.generator_optimizer =     keras.optimizers.Adam(learning_rate=GEN_LR)
        self.discriminator_optimizer = keras.optimizers.Adam(learning_rate=DISC_LR)

        self.history = {
            "discriminator_loss": [],
            "discriminator_accuracy": [],
            "generator_loss": [],
            "generator_accuracy": []
        }
    
    def adjust_lr(self, avg_disc_loss, avg_gen_loss, tolerance=0.1, min_lr=1e-8):
        """
        Ajusta dinamicamente o LR com base na razão nas métricas de perda.
        Args:
            avg_disc_loss (float): Perda média do discriminador.
            avg_gen_loss (float): Perda média do gerador.
            tolerance (float): Margem de tolerância para o ajuste.
            min_lr (float): Valor mínimo permitido para o learning rate.
        """
        gen_lr = self.generator_optimizer.learning_rate.numpy()
        disc_lr = self.discriminator_optimizer.learning_rate.numpy()
        # Calcula a razão entre as perdas
        ratio = avg_gen_loss / (avg_disc_loss + 1e-8)  # Evita divisão por zero

        # Ajusta apenas se a razão estiver fora da margem de tolerância
        if ratio > (1 + tolerance):  # Discriminador está forte
            new_disc_lr = max(min_lr, disc_lr * 0.9)  # Reduz LR do discriminador
            self.discriminator_optimizer.learning_rate.assign(new_disc_lr)
            print(f"Reduzindo LR do discriminador: {disc_lr} -> {new_disc_lr}")

        elif ratio < (1 - tolerance):  # Gerador está forte
            new_gen_lr = max(min_lr, gen_lr * 0.9)  # Reduz LR do gerador
            self.generator_optimizer.learning_rate.assign(new_gen_lr)
            print(f"Reduzindo LR do gerador: {gen_lr} -> {new_gen_lr}")

        else:
            # Ambos os modelos estão equilibrados, pode-se optar por reduzir suavemente os LRs
            new_gen_lr = gen_lr * .95
            new_disc_lr = disc_lr * .95

            self.generator_optimizer.learning_rate.assign(new_gen_lr)
            self.discriminator_optimizer.learning_rate.assign(new_disc_lr)
            print(f"Reduzindo LR de ambos: Gerador = {new_gen_lr}, Discriminador = {new_disc_lr}")

    # loss funcs
    def generator_loss(self, fake_output):
        """
        Retorna o valor de perda do gerador em um batch.
        Args:
            fake_output (tf.Tensor): Imagens feitas pelo gerador
        """
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        """
        Retorna o valor de perda do discriminador em um batch.
        Args:
            fake_output (tf.Tensor): Imagens feitas pelo gerador.
            real_output (tf.Tensor): Imagens reais do dataset.
        """
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = float(real_loss) + float(fake_loss)
        return total_loss

    @tf.function
    def disc_acc(self, t_preds, f_preds=None):
        """
        Cálculo da métrica de acurácia do discriminador.
        Args:
            t_preds (tf.Tensor): Previsões do discriminador das imagens reais.
            f_preds (tf.Tensor): Previsões do discriminador das imagens feitas pelo gerador.
        """
        preds = t_preds if f_preds is None else tf.concat([f_preds, t_preds], axis=0)

        preds = tf.greater(preds, .5)
        preds = tf.cast(preds, tf.float32)

        return tf.reduce_mean(preds)

    @tf.function
    def gen_acc(self, preds):
        """
        Cálculo da métrica de acurácia do gerador.
        Args:
            preds (tf.Tensor): Previsões do discriminador das imagens feitas pelo gerador.
        """
        preds = tf.greater(preds, .5)
        preds = tf.cast(preds, tf.float16)
    
        # Normalizando a acurácia e retornando como float
        return tf.reduce_mean(preds)
    
    def catch_losses_mean(self, epochs_num=1, verbose=None):
        """
        Retorna a média de loss das últimas {epochs_num} épocas.
        Args:
            epochs_num (int): Número de épocas para calcular a média
            verbose (bool): Exibe as médias obtidas na saída stdout.
        """
        disc_losses = self.history["discriminator_loss"][-epochs_num:]
        gen_losses  = self.history["generator_loss"][-epochs_num:]
    
        if not disc_losses or not gen_losses:
            warnings.warn("AVISO: Histórico vazio ou insuficiente para o número de perdas solicitado.")

        # Calculo das as médias
        avg_disc_loss = np.mean(disc_losses)
        avg_gen_loss  = np.mean(gen_losses)
        
        if verbose:
            print(f"Média de perda do discriminador: {avg_disc_loss:.5f}")
            print(f"Média de perda do gerador: {avg_gen_loss:.5f}")

        return avg_disc_loss, avg_gen_loss

    def generate_and_save_images(self, epoch, num_images=9, path=None, name='generated_image.png', verbose=None):
        """
        Salva uma imagem com as amostras feitas pelo gerador.
        Args:
            epoch (int): Número da época para salvar no nome da imagem gerada.
            num_images (int): Número de amostras que serão geradas.
            path (str): Diretório para salvar a imagem.
            name (str): Nome do arquivo a ser salvo.
            verbose (int): Exibe a imagem e o caminho onde foi salva.
        """
        path = os.path.join(path, 'generated_images') if path else os.path.join('generated_images')
        os.makedirs(path, exist_ok=True)
        
        fig_name = f"image_at_epoch_{epoch}.png" if epoch else name
        fig_name = os.path.join(path, fig_name)
        
        # Gerando as imagens
        noise = tf.random.normal([num_images, self.NOISE_DIM])
        generated_images = self.generator(noise, training=False)

        # 'Desnormalização' das imagens geradas [-1, 1] -> [0, 255]
        generated_images = (generated_images + 1) * 127.5
        generated_images = np.clip(generated_images, 0, 255)
        generated_images = generated_images.astype(np.uint8)

        # Tamanho da grade
        grid_size = int(np.sqrt(num_images))
        plt.figure(figsize=(grid_size, grid_size))

        # Adiciona as imagens à grade
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                plt.subplot(grid_size, grid_size, idx + 1)
                plt.imshow(generated_images[idx])
                plt.axis('off')

        plt.savefig(fig_name, dpi=300, bbox_inches='tight', pad_inches=0)
        if verbose:
            plt.show()
        
        if verbose:
          print('Imagem salva em: ', path)

    def save_model(self, epoch=1, path=None, metrics=None, verbose=None):
        """
        Serializa e salva ambos os modelos (gerador e discriminador) no formato .keras

        Args:
            epoch (int): Inserido nos nomes dos arquivos para identificar a época em que foram salvos.
            path (str): Diretório onde os arquivos serão salvos.
            metrics (bool): Salva as métricas do modelo.
            verbose (bool): Exibe as informações dos arquivos salvos.
        """
        path = path if path else 'checkpoint'
        os.makedirs(path, exist_ok=True)
        
        name = lambda s,x: f"{s}_at_epoch_{x}.keras"
        generator_name = os.path.join(path, name('generator', epoch))
        discriminator_name = os.path.join(path, name('discriminator', epoch))

        self.discriminator.save(discriminator_name)
        if verbose:
            print(f"Discriminador salvo em: {discriminator_name}")

        self.generator.save(generator_name)
        if verbose:    
            print(f"Gerador salvo em: {discriminator_name}")
        if metrics:
            metrics_name = os.path.join(path, name('metrics', epoch))
            with open(metrics_name, 'w') as f:
                json.dump(self.history, f)

    def load_model(self, discriminator_path, generator_path, verbose=None):
        """
        Restaura ambos os modelos (gerador e discriminador) carregando seus respectivos arquivos .keras
        Args:
            discriminator_path (str): Endereço do arquivo do discriminador.
            generator_path (str): Endereço do arquivo do gerador.
            verbose (bool): Exibe as informações de carregamento.
        """
        self.generator = keras.models.load_model(generator_path)
        if verbose:
            print("Gerador carregado.")
        self.discriminator= keras.models.load_model(discriminator_path)
        if verbose:
            print("Discriminador carregado.")
        
    @tf.function
    def train_step(self, images, extra_step=None):
        """
        Processo de um passo do treinamento da GAN.
        Args:
            images (tensor): Conjunto de imagens (batch) do Dataset.
            extra_step (bool): Ativa um passo de treinamento extra para o discriminador com o dataset real.
        """
        noise = tf.random.normal(shape=(self.BATCH_SIZE, self.NOISE_DIM))
        gen_loss, disc_loss = None, None
        gen_acc, disc_acc = None, None

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_imgs = self.generator(noise, training=True)

            # predições do discriminador (com imagens reais e sintéticas)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(fake_imgs, training=True)

            # Tentar calcular e aplicar o gradiente novamente pode ser melhor
            if extra_step:
                real_output = tf.concat([real_output, self.discriminator(images, training=True)], axis=0)

            # calculando as métricas de loss
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            
            # calculando as métricas de acurácia
            disc_acc = self.disc_acc(real_output, fake_output)
            gen_acc = self.gen_acc(fake_output)

        # Salvando e atualizando os gradientes
        generator_grads = gen_tape.gradient(gen_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(generator_grads, self.generator.trainable_weights))

        discriminator_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_weights))

        return disc_loss, disc_acc, gen_loss, gen_acc

    def train(self, dataset, epochs=None, epoch_to_adjust_lr=5, epochs_checkpoint=10, path=None, epochs_to_extra_step=2):
        """
        Unifica e executa todos os processos necessários para o treinamento da GAN.
        Args:
            dataset (tf.data.Dataset): Imagens para o treinamento.
            epochs (int): Número de vezes que o modelo passará por todo o dataset.
            epoch_to_adjust_lr (int): Número de épocas para percorrer antes de ajustar o lr.
            epochs_checkpoint (int): Número de épocas até salvar um Checkpoint.
            path (str): Caminho para salvar os arquivos gerados durante o treinamento
        """
        epochs = epochs if epochs else self.EPOCHS
        
        for epoch in range(epochs):
            start = time.time()
            epoch_dloss, epoch_dacc, epoch_gloss, epoch_gacc = [], [], [], []
            
            for image_batch in dataset:
                disc_loss, disc_acc, gen_loss, gen_acc = (self.train_step(image_batch, extra_step=True)
                                                          if not epoch%epochs_to_extra_step
                                                          else
                                                          self.train_step(image_batch, extra_step=False))
                epoch_dloss.append(np.mean(disc_loss))
                epoch_dacc.append(np.mean(disc_acc))
                epoch_gloss.append(np.mean(gen_loss))
                epoch_gacc.append(np.mean(gen_acc))
                
                sys.stdout.write(f"\rGenerator loss: {gen_loss:.5f}\tDiscriminator loss: {disc_loss:.5f}\tGenerator accuracy: {gen_acc:.5f}\tDiscriminator accuracy:{disc_acc:.5f}")
                
            epoch_dloss = np.mean(epoch_dloss)
            epoch_dacc = np.mean(epoch_dacc)
            epoch_gloss = np.mean(epoch_gloss)
            epoch_gacc = np.mean(epoch_gacc)
            
            self.history["discriminator_loss"].append(epoch_dloss)
            self.history["discriminator_accuracy"].append(epoch_dacc)
            self.history["generator_loss"].append(epoch_gloss)
            self.history["generator_accuracy"].append(epoch_gacc)
                
            display.clear_output(wait=True)
            self.generate_and_save_images(epoch + 1, path=path, verbose=True)
            
            # salva o modelo no atual estado
            if not (epoch + 1) % epochs_checkpoint:
                self.save_model(epoch=epoch+1, verbose=True, path=path)

            # Ajuste de lr
            if not (epoch + 1) % epoch_to_adjust_lr:
                disc_loss, gen_loss = self.catch_losses_mean(epochs_num=epoch_to_adjust_lr, verbose=True)
                self.adjust_lr(avg_disc_loss=disc_loss, avg_gen_loss=gen_loss)

            epoch_time = time.time()-start
            print(f"Tempo para a época {epoch+1}: {epoch_time:.2f}s.")
            print(f"Generator loss: {epoch_gloss:.4f}\tDiscriminator loss: {epoch_dloss:.4f}\t")
            
            estimated_time = epoch_time * (epochs - epoch)
            sys.stdout.write(f"Tempo estimado para conclusão do treinamento: ~{round(estimated_time/60)}min\n")
            
            # barra de progresso simples
            x = int(np.round((100 * (epochs - (epoch + 1))) / epochs))
            sys.stdout.write(f"\r[{'█' * (100-x)}{'-' * x}]\n")
        
        # Gera imagens para cada época
        display.clear_output(wait=True)
        self.generate_and_save_images(epoch + 1)