import torch
import torch.nn as nn

class MDTA(nn.Module):
    def __init__(self , channels , num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.W_p = nn.Conv2d(channels , channels * 3 , kernel_size = 1 , bias = False)
        self.W_d = nn.Conv2d(channels * 3 , channels * 3 , kernel_size = 3 , groups = channels * 3 , padding = 1 , bias = False) #DepthWise Conv
        self.out = nn.Conv2d(channels , channels , kernel_size = 1  , bias = False)

    def forward(self , inp):
        batch , channels , height , width = inp.shape
        weights_p  = self.W_p(inp)
        weights_d = self.W_d(weights_p)
        query , key , value = weights_d.chunk(3 , dim = 1)

        query = query.reshape(batch , self.num_heads , -1 , height * width)
        query = torch.nn.functional.normalize(query , dim = -1)
        key = key.reshape(batch , self.num_heads , -1 , height * width)
        key = torch.nn.functional.normalize(key , dim = -1)
        value = value.reshape(batch , self.num_heads , -1 , height * width)


        transposed_attention = torch.matmul(query , key.transpose(-2, -1))
        transposed_attention = nn.Softmax(dim = -1)(transposed_attention)
        # print(value.size() , transposed_attention.size())
        out = torch.matmul(transposed_attention , value).reshape(batch , -1 , height , width)
        out = self.out(out)
        return out
    
class GDFN(nn.Module):
    def __init__(self , channels , expansion_factor):
        super(GDFN, self).__init__()
        self.expanded_features = int(channels * expansion_factor)
        self.W_p = nn.Conv2d(channels , self.expanded_features * 2 , kernel_size = 1 , bias = False)
        self.W_d = nn.Conv2d(self.expanded_features * 2 , self.expanded_features * 2 , kernel_size = 3 , groups = self.expanded_features * 2 , padding = 1 , bias = False)
        self.out = nn.Conv2d(self.expanded_features , channels , kernel_size = 1 , bias = False)

    def forward(self , inp):
        weights_p = self.W_p(inp)
        weights_d = self.W_d(weights_p)
        non_linear , linear = weights_d.chunk(2 , dim = 1)
        non_linear = torch.nn.functional.gelu(non_linear)
        out = non_linear * linear
        out = self.out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, num_heads , num_channels , expansion_factor):
        super(TransformerBlock, self).__init__()
        self.channels = num_channels
        self.MDTA_block = MDTA(num_channels , num_heads)
        self.GDFN_block = GDFN(num_channels , expansion_factor)
        self.norm1 = nn.LayerNorm(self.channels)
        self.norm2 = nn.LayerNorm(self.channels)

    def forward(self , inp):
        batch , channels , height , width = inp.shape
        # Layer_norm{acrross_channels} -> MDTA
        inp_norm = self.norm1(inp.reshape(batch , channels , -1).transpose(-2 , -1))
        inp_norm = inp_norm.transpose(-2 , -1).reshape(batch , channels , height , width)
        mdta_out = self.MDTA_block(inp_norm) + inp

        #Layer_norm{acrross channels}(mdta_output) -> GDFN
        mdta_norm = self.norm2(mdta_out.reshape(batch , channels , -1).transpose(-2 , -1))
        mdta_norm = mdta_norm.transpose(-2, -1).reshape(batch , channels , height , width)
        gdfn_out = self.GDFN_block(mdta_norm) + mdta_out
        return gdfn_out

class Upsample(nn.Module):
    def __init__(self , num_channels , upscale_factor = 2):
        super(Upsample, self).__init__()
        self.upscale_conv = nn.Conv2d(num_channels , num_channels * upscale_factor , kernel_size = 3 , padding = 1 , bias = False)
        self.upscale = nn.PixelShuffle(upscale_factor)

    def forward(self , inp):
        upscale_out = self.upscale_conv(inp)
        upscale_out = self.upscale(upscale_out)
        return upscale_out
    
class Downsample(nn.Module):
    def __init__(self , num_channels , downscale_factor = 2):
        super(Downsample, self).__init__()
        self.downscale_conv = nn.Conv2d(num_channels , num_channels // downscale_factor , kernel_size = 3 , padding = 1 , bias = False)
        self.downscale = nn.PixelUnshuffle(downscale_factor)

    def forward(self , inp):
        downscale_out = self.downscale_conv(inp)
        downscale_out = self.downscale(downscale_out)
        return downscale_out
    
class Restormer(nn.Module):
    def __init__(self ,
                 input_dim = 3 ,
                 out_dim = 3,
                 embed_dim = 48 ,
                 num_heads = [1,2,4,8] , 
                 num_channels = [48 , 96 , 192 , 384] , 
                 num_tf_blocks = [4, 6, 6, 8] ,
                 num_refinement_blocks = 4 ,
                 expansion_factor = 2.66
                ):

        super(Restormer, self).__init__()
        self.conv_embed = nn.Conv2d(input_dim , embed_dim , kernel_size = 3 , padding = 1 , bias = False)

        tf_blocks_1 = []
        for i in range(num_tf_blocks[0]):
            tf_blocks_1.append(TransformerBlock(num_heads[0] , num_channels[0] , expansion_factor = expansion_factor))
        self.encoder_1 = nn.Sequential(*tf_blocks_1)

        self.downsample_1_2 = Downsample(num_channels[0])
        tf_blocks_2 = []
        for i in range(num_tf_blocks[1]):
            tf_blocks_2.append(TransformerBlock(num_heads[1] , num_channels[1] , expansion_factor = expansion_factor))
        self.encoder_2 = nn.Sequential(*tf_blocks_2)

        self.downsample_2_3 = Downsample(num_channels[1])
        tf_blocks_3 = []
        for i in range(num_tf_blocks[2]):
            tf_blocks_3.append(TransformerBlock(num_heads[2] , num_channels[2] , expansion_factor = expansion_factor))
        self.encoder_3 = nn.Sequential(*tf_blocks_3)

        self.downsample_3_4 = Downsample(num_channels[2])
        tf_blocks_4 = []
        for i in range(num_tf_blocks[3]):
            tf_blocks_4.append(TransformerBlock(num_heads[3] , num_channels[3] , expansion_factor = expansion_factor))
        self.encoder_4 = nn.Sequential(*tf_blocks_4)

        self.upsample_4_3 = Upsample(num_channels[3])
        self.channel_reduction_1 = nn.Conv2d(num_channels[3] , num_channels[2] , kernel_size = 1 , bias = False)
        tf_blocks_5 = []
        for i in range(num_tf_blocks[2]):
            tf_blocks_5.append(TransformerBlock(num_heads[2] , num_channels[2] , expansion_factor = expansion_factor))
        self.decoder_3 = nn.Sequential(*tf_blocks_5)

        self.upsample_3_2 = Upsample(num_channels[2])
        self.channel_reduction_2 = nn.Conv2d(num_channels[2] , num_channels[1] , kernel_size = 1 , bias = False)
        tf_blocks_6 = []
        for i in range(num_tf_blocks[1]):
            tf_blocks_6.append(TransformerBlock(num_heads[1] , num_channels[1] , expansion_factor = expansion_factor))
        self.decoder_2 = nn.Sequential(*tf_blocks_6)

        self.upsample_2_1 = Upsample(num_channels[1])
        tf_blocks_7 = []
        for i in range(num_tf_blocks[0]):
            tf_blocks_7.append(TransformerBlock(num_heads[0] , num_channels[1] , expansion_factor = expansion_factor))
        self.decoder_1 = nn.Sequential(*tf_blocks_7)

        tf_refinement_blocks = []
        for i in range(num_refinement_blocks):
            tf_refinement_blocks.append(TransformerBlock(num_heads[0] , num_channels[1] , expansion_factor = expansion_factor))
        self.refinement_block = nn.Sequential(*tf_refinement_blocks)
        self.out_conv = nn.Conv2d(num_channels[1] , out_dim , kernel_size = 3 , padding = 1 , bias = False)

    def forward(self, inp):
        feat_embed_0 = self.conv_embed(inp)
        encoder_1 = self.encoder_1(feat_embed_0)

        downsample_1 = self.downsample_1_2(encoder_1)
        encoder_2 = self.encoder_2(downsample_1)

        downsample_2 = self.downsample_2_3(encoder_2)
        encoder_3 = self.encoder_3(downsample_2)

        downsample_3 = self.downsample_3_4(encoder_3)
        encoder_4 = self.encoder_4(downsample_3)

        upsample_3 = self.upsample_4_3(encoder_4)
        skip_connection_3 = torch.cat([encoder_3 , upsample_3] , dim = 1)
        channel_reduce_3 = self.channel_reduction_1(skip_connection_3)
        decoder_3 = self.decoder_3(channel_reduce_3)

        upsample_2 = self.upsample_3_2(decoder_3)
        skip_connection_2 = torch.cat([encoder_2 , upsample_2] ,dim = 1)
        channel_reduce_2 = self.channel_reduction_2(skip_connection_2)
        decoder_2 = self.decoder_2(channel_reduce_2)

        upsample_1 = self.upsample_2_1(decoder_2)
        skip_connection_1 = torch.cat([encoder_1 , upsample_1] , dim = 1)
        decoder_1 = self.decoder_1(skip_connection_1)

        refinement = self.refinement_block(decoder_1)
        output = self.out_conv(refinement)
        output = output + inp
        return output
    





    

    


    

        
    





