import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch_mobility.metrics.collective import common_part_of_commuters
from torch_mobility.models.layers.module import get_activation_function

from torch_mobility.models.layers.RLE import TheoryGridCellSpatialRelationEncoder,MultiLayerFeedForwardNN
from torch_mobility.models.layers.layers import DropPath, TransformerEncoderLayer

class TransFlower(pl.LightningModule):
    def __init__(self, args):
        
        super(TransFlower, self).__init__()
        self.save_hyperparameters(args)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_cpc = []
        self.test_step_rmse = []
        self.test_step_mae = []

        self.bn_loc = args.bn_loc
        self.loc_combine = args.loc_combine
        self.learning_rate = args.lr
        self.input_size = args.input_size
        self.n_half_size_layers = args.n_half_size_layers
        self.opt = args.opt
        self.momentum = args.momentum
        
        self.linear_in = torch.nn.Linear(args.input_size, args.hidden_size)

        self.linear_full_to_half = torch.nn.Linear(args.hidden_size, args.hidden_size // 2)
        self.linear_half = torch.nn.Linear(args.hidden_size // 2, args.hidden_size // 2)
        self.linear_out = torch.nn.Linear(args.hidden_size // 2, args.output_size)

        self.act = get_activation_function(args.act, "encoder")

        self.dropout = torch.nn.Dropout(args.p)
        if args.bn_loc:
            if self.loc_combine == 'none':
                ffn_sp_en = MultiLayerFeedForwardNN(input_dim=int(6*args.freq),output_dim=args.hidden_size, num_hidden_layers=args.num_hidden_layer,dropout_rate=args.p,
                                                hidden_dim=args.hidden_dim,use_layernormalize=args.use_layn,skip_connection=args.skip_connection,
                                                activation=args.ffn_f_act)
                self.location_encoder = TheoryGridCellSpatialRelationEncoder(spa_embed_dim=args.hidden_size, max_radius= args.max_radius,
                                                                        min_radius=args.min_radius, frequency_num=args.freq, freq_init=args.freq_init,ffn= ffn_sp_en,uv_theta=0)
            elif self.loc_combine == 'cat':
                ffn_sp_en1 = MultiLayerFeedForwardNN(input_dim=int(6*args.freq),output_dim=args.hidden_size//2, num_hidden_layers=args.num_hidden_layer,dropout_rate=args.p,
                                                hidden_dim=args.hidden_dim,use_layernormalize=args.use_layn,skip_connection=args.skip_connection,
                                                activation=args.ffn_f_act)
                ffn_sp_en2 = MultiLayerFeedForwardNN(input_dim=int(6*args.freq),output_dim=args.hidden_size//2, num_hidden_layers=args.num_hidden_layer,dropout_rate=args.p,
                                                hidden_dim=args.hidden_dim,use_layernormalize=args.use_layn,skip_connection=args.skip_connection,
                                                activation=args.ffn_f_act)
                self.location_encoder1 = TheoryGridCellSpatialRelationEncoder(spa_embed_dim=args.hidden_size//2, max_radius= args.max_radius,
                                                                        min_radius=args.min_radius, frequency_num=args.freq, freq_init=args.freq_init,ffn= ffn_sp_en1,uv_theta=0)
                self.location_encoder2 = TheoryGridCellSpatialRelationEncoder(spa_embed_dim=args.hidden_size//2, max_radius= args.max_radius,
                                                                        min_radius=args.min_radius, frequency_num=args.freq, freq_init=args.freq_init,ffn= ffn_sp_en2,uv_theta=30)
            else:
                ffn_sp_en1 = MultiLayerFeedForwardNN(input_dim=int(6*args.freq),output_dim=args.hidden_size, num_hidden_layers=args.num_hidden_layer,dropout_rate=args.p,
                                                hidden_dim=args.hidden_dim,use_layernormalize=args.use_layn,skip_connection=args.skip_connection,
                                                activation=args.ffn_f_act)
                ffn_sp_en2 = MultiLayerFeedForwardNN(input_dim=int(6*args.freq),output_dim=args.hidden_size, num_hidden_layers=args.num_hidden_layer,dropout_rate=args.p,
                                                hidden_dim=args.hidden_dim,use_layernormalize=args.use_layn,skip_connection=args.skip_connection,
                                                activation=args.ffn_f_act)
                self.location_encoder1 = TheoryGridCellSpatialRelationEncoder(spa_embed_dim=args.hidden_size, max_radius= args.max_radius,
                                                                            min_radius=args.min_radius, frequency_num=args.freq, freq_init=args.freq_init,ffn= ffn_sp_en1,uv_theta=0)
                self.location_encoder2 = TheoryGridCellSpatialRelationEncoder(spa_embed_dim=args.hidden_size, max_radius= args.max_radius,
                                                                            min_radius=args.min_radius, frequency_num=args.freq, freq_init=args.freq_init,ffn= ffn_sp_en2,uv_theta=30)
                if self.loc_combine == 'ffn':
                    self.fusion_ffn = MultiLayerFeedForwardNN(input_dim=args.hidden_size*2,output_dim=args.hidden_size, 
                                                                num_hidden_layers=args.comb_num_hidden_layer,dropout_rate=args.comb_p,
                                                                hidden_dim=args.comb_hidden_dim,use_layernormalize=args.comb_use_layn,
                                                                skip_connection=args.comb_skip_connection,
                                                                activation=args.comb_ffn_f_act
                                                                )
        
        self.droppath = DropPath(args.dp_path)
        self.trans_encoders = nn.ModuleList([TransformerEncoderLayer(args.hidden_size, 
                                            heads=args.n_heads, p=args.tr_p,act_attention=args.mh_act) for _ in range(args.encoder_num)])

        self.norm_enc = nn.LayerNorm(args.hidden_size)


    def forward(self, x):

        x = self.linear_in(x)
        x = self.act(x)
        x = self.dropout(x)

        for trans_enc in self.trans_encoders:
            x = trans_enc(x)
            x = self.droppath(x)

        x = self.norm_enc(x)
        
        x = self.linear_full_to_half(x)
        x = self.act(x)
        x = self.dropout(x)

        for i in range(self.n_half_size_layers):
            x = self.linear_half(x)
            x = self.act(x)
            x = self.dropout(x)

        x = self.linear_out(x)

        return x

    def forward1(self, x, od_loc):

        x = self.linear_in(x)
        x = self.act(x)
        x = self.dropout(x)

        if self.loc_combine == 'none':
            pe = self.location_encoder(od_loc).to(x.device)
        else:
            pe1 = self.location_encoder1(od_loc).to(x.device)
            pe2 = self.location_encoder2(od_loc).to(x.device)
            if self.loc_combine == 'cat':
                pe = torch.cat([pe1,pe2],dim=-1)
            elif self.loc_combine == 'avg':
                pe = (pe1+pe2)/2
            elif self.loc_combine == 'ffn':
                pe = torch.cat([pe1,pe2],dim=-1)
                pe = self.fusion_ffn(pe)
        x = x + pe

        for trans_enc in self.trans_encoders:
            x = trans_enc(x)
            x = self.droppath(x)

        x = self.norm_enc(x)

        x = self.linear_full_to_half(x)
        x = self.act(x)
        x = self.dropout(x)

        for i in range(self.n_half_size_layers):
            x = self.linear_half(x)
            x = self.act(x)
            x = self.dropout(x)

        x = self.linear_out(x)

        return x
    

    def training_step(self, batch, batch_idx):
        if self.bn_loc:
            x, y, origin_loc,  destination_loc = batch
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            od_loc = destination_loc - origin_loc

            y_hat = self.forward1(x, od_loc)

        else:
            x, y = batch
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            y_hat = self.forward(x)

        y_hat = y_hat.view(y_hat.shape[0], y_hat.shape[1]) #[bs, num_dest]

        loss = self.loss(y_hat, y)
        self.training_step_outputs.append(loss)
        self.log('train_loss', loss)
        return loss
    
    def on_train_epoch_end(self):
        loss = torch.stack(self.training_step_outputs).mean()
        self.log("train_loss", loss)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        sm_layer = torch.nn.Softmax(dim=1)
        if self.bn_loc:
            x, y, origin_loc,  destination_loc = batch
            x = torch.tensor(x, dtype=torch.float32) 
            y = torch.tensor(y, dtype=torch.float32)

            od_loc = destination_loc - origin_loc
            y_hat = sm_layer(self.forward1(x, od_loc))
        else:
            x, y = batch
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            y_hat = sm_layer(self.forward(x))

        loss = self.loss(y_hat.view(y_hat.shape[0], y_hat.shape[1]), y)

        total_out_trips = torch.sum(y, dim=-1)

        model_od = (y_hat.T * total_out_trips).T
        model_od = model_od.view(model_od.shape[0], model_od.shape[1])

        cpc = common_part_of_commuters(y, model_od)
        self.validation_step_outputs.append(cpc)
        self.log('val_loss', loss)
        self.log('val_cpc', cpc)

        return cpc
    
    def on_validation_epoch_end(self):
        cpc = torch.stack(self.validation_step_outputs).mean()
        self.log("val_cpc", cpc)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        sm_layer = torch.nn.Softmax(dim=1)
        if self.bn_loc:
            x, y, origin_loc,  destination_loc = batch
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            od_loc = destination_loc - origin_loc
            y_hat = sm_layer(self.forward1(x, od_loc))
        else:
            x, y = batch
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            y_hat = sm_layer(self.forward(x))

        if y_hat.shape[1] == 1:
            self.model_od = None
            return

        loss = self.loss(y_hat.view(y_hat.shape[0], y_hat.shape[1]), y)

        total_out_trips = torch.sum(y, dim=-1)

        model_od = (y_hat.T * total_out_trips).T
        model_od = model_od.view(model_od.shape[0], model_od.shape[1])
        self.model_od = model_od

        cpc = common_part_of_commuters(y, model_od)
        rmse = torch.sqrt(torch.mean((y - model_od)**2))
        mae = torch.mean(torch.abs(y - model_od))

        self.test_step_cpc.append(cpc)
        self.test_step_rmse.append(rmse)
        self.test_step_mae.append(mae)

        return cpc


    def on_test_epoch_end(self):
        cpc = torch.stack(self.test_step_cpc).mean()
        rmse = torch.stack(self.test_step_rmse).mean()
        mae = torch.stack(self.test_step_mae).mean()

        self.log("test_cpc", cpc)
        self.log("test_rmse", rmse)
        self.log("test_mae", mae)
        self.test_step_cpc.clear()
        self.test_step_rmse.clear()
        self.test_step_mae.clear()

    def configure_optimizers(self):
        if self.opt == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.opt == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif self.opt == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        return optimizer

    def evaluation(self, y_hat, y):
        return None

    def loss(self, y_hat, y):
        lsm = torch.nn.LogSoftmax(dim=1)
        return -(y * lsm(torch.squeeze(y_hat, dim=-1))).sum()

    def optimizer(self):
        return None

