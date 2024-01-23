import torch
from torch import nn, optim

class LSD_loss( nn.Module ):
    def __init__( self ):
        super( LSD_loss, self ).__init__()

    def forward( self, y_true, y_pred ):
        LSD = torch.mean( ( y_true - y_pred ) ** 2, dim=0 )
        LSD = torch.mean( torch.sqrt( LSD ), dim=1 )
        LSD = torch.mean( LSD )
        
        return LSD
    

class UNet( nn.Module ):
    def __init__( self, ):
        super( UNet, self ).__init__()

        self.e1 = nn.Conv2d( 1, 64, kernel_size=( 7, 5 ), stride=( 2, 1 ), padding=( 3, 2 ) )
        nn.init.normal_( self.e1.weight, mean=0, std=0.2 )
        self.bn1 = nn.BatchNorm2d( 64 )
        self.lrelu = nn.LeakyReLU( 0.2 )
        # 128 16

        self.e2 = nn.Conv2d( 64, 128, kernel_size=( 7, 5 ), stride=( 2, 1 ), padding=( 3, 2 ) )
        nn.init.normal_( self.e2.weight, mean=0, std=0.2 )
        self.bn2 = nn.BatchNorm2d( 128 )
        # 64 16

        self.e3 = nn.Conv2d( 128, 256, kernel_size=( 7, 5 ), stride=( 2, 1 ), padding=( 3, 2 ) )
        nn.init.normal_( self.e3.weight, mean=0, std=0.2 )
        self.bn3 = nn.BatchNorm2d( 256 )
        # 32 16

        self.e4 = nn.Conv2d( 256, 512, kernel_size=( 5, 5 ), stride=( 2, 1 ), padding=( 2, 2 ) )
        nn.init.normal_( self.e4.weight, mean=0, std=0.2 )
        self.bn4 = nn.BatchNorm2d( 512 )
        # 16 16

        self.e5 = nn.Conv2d( 512, 512, kernel_size=( 5, 5 ), stride=( 2, 2 ), padding=( 2, 2 ) )
        nn.init.normal_( self.e5.weight, mean=0, std=0.2 )
        self.bn5 = nn.BatchNorm2d( 512 )
        # 8 8

        self.e6 = nn.Conv2d( 512, 512, kernel_size=( 3, 3 ), stride=( 2, 2 ), padding=( 1, 1 ) )
        nn.init.normal_( self.e6.weight, mean=0, std=0.2 )
        self.bn6 = nn.BatchNorm2d( 512 )
        # 4 4

        self.e7 = nn.Conv2d( 512, 512, kernel_size=( 3, 3 ), stride=( 2, 2 ), padding=( 1, 1 ) )
        nn.init.normal_( self.e7.weight, mean=0, std=0.2 )
        self.bn7 = nn.BatchNorm2d( 512 )

        self.e8 = nn.Conv2d( 512, 512, kernel_size=( 3, 3 ), stride=( 2, 2 ), padding=( 1, 1 ) )
        nn.init.normal_( self.e8.weight, mean=0, std=0.2 )
        self.bn8 = nn.BatchNorm2d( 512 )

        # 2 x 2
        self.upsample2x = nn.Upsample( scale_factor=2, mode="bilinear", align_corners=True )  
        self.bn9_1 = nn.BatchNorm2d( 512 )
        self.d1 = nn.Conv2d( 1024, 512, kernel_size=( 3, 3 ), padding="same" )
        nn.init.normal_( self.d1.weight, mean=0, std=0.2 )
        self.bn9_2  = nn.BatchNorm2d( 512 )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout( 0.5 )

        # 4 x 4
        self.bn10_1 = nn.BatchNorm2d( 512 )  
        self.d2 = nn.Conv2d( 1024, 512, kernel_size=( 3, 3 ), padding="same" )
        nn.init.normal_( self.d2.weight, mean=0, std=0.2 )
        self.bn10_2  = nn.BatchNorm2d( 512 )

        # 8 x 8
        self.bn11_1 = nn.BatchNorm2d( 512 )  
        self.d3 = nn.Conv2d( 1024, 512, kernel_size=( 3, 3 ), padding="same" )
        nn.init.normal_( self.d3.weight, mean=0, std=0.2 )
        self.bn11_2  = nn.BatchNorm2d( 512 )

        # 16 x 16
        self.bn12_1 = nn.BatchNorm2d( 512 )  
        self.d4 = nn.Conv2d( 1024, 256, kernel_size=( 5, 5 ), padding="same" )
        nn.init.normal_( self.d4.weight, mean=0, std=0.2 )
        self.bn12_2  = nn.BatchNorm2d( 256 )

        # 32 x 16
        self.upsample32 = nn.Upsample( size=( 32, 16 ), mode="bilinear", align_corners=True )
        self.bn13_1 = nn.BatchNorm2d( 256 )
        self.d5 = nn.Conv2d( 512, 128, kernel_size=( 5, 5 ), padding="same" )
        nn.init.normal_( self.d5.weight, mean=0, std=0.2 )
        self.bn13_2  = nn.BatchNorm2d( 128 )

        # 64 x 16
        self.upsample64 = nn.Upsample( size=( 64, 16 ), mode="bilinear", align_corners=True )
        self.bn14_1 = nn.BatchNorm2d( 128 )
        self.d6 = nn.Conv2d( 256, 64, kernel_size=( 7, 5 ), padding="same" )
        nn.init.normal_( self.d6.weight, mean=0, std=0.2 )
        self.bn14_2  = nn.BatchNorm2d( 64 )

        # 128 x 16
        self.upsample128 = nn.Upsample( size=( 128, 16 ), mode="bilinear", align_corners=True )
        self.bn15_1 = nn.BatchNorm2d( 64 )
        self.d7 = nn.Conv2d( 128, 32, kernel_size=( 7, 5 ), padding="same" )
        nn.init.normal_( self.d7.weight, mean=0, std=0.2 )
        self.bn15_2  = nn.BatchNorm2d( 32 )

        # 256 x 16
        self.upsample256 = nn.Upsample( size=( 256, 16 ), mode="bilinear", align_corners=True )
        self.bn16_1 = nn.BatchNorm2d( 32 )
        self.output = nn.Conv2d( 32, 1, kernel_size=( 7, 5 ), padding="same" )
        nn.init.normal_( self.output.weight, mean=0, std=0.2 )
        self.bn16_2  = nn.BatchNorm2d( 1 )
        self.softmax = nn.Softmax()
        # self.output = nn.ConvTranspose2d( 128, 1, kernel_size=( 5, 7 ), stride=( 1, 2 ) )

    def forward( self, x ):
        # Input layer
        x = x.unsqueeze( dim=1 )
        # print(x.shape)

        # Encoder
        x = self.e1( x )
        x = self.bn1( x )
        x = self.lrelu( x )
        x1 = x
        # print(x.shape)

        x = self.e2( x )
        x = self.bn2( x )
        x = self.lrelu( x )
        x2 = x
        # print(x.shape)

        x = self.e3( x )
        x = self.bn3( x )
        x = self.lrelu( x )
        x3 = x
        # print(x.shape)

        x = self.e4( x )
        x = self.bn4( x )
        x = self.lrelu( x )
        x4 = x
        # print(x.shape)

        x = self.e5( x )
        x = self.bn5( x )
        x = self.lrelu( x )
        x5 = x
        # print(x.shape)

        x = self.e6( x )
        x = self.bn6( x )
        x = self.lrelu( x )
        x6 = x
        # print(x.shape)

        x = self.e7( x )
        x = self.bn7( x )
        x = self.lrelu( x )
        x7 = x
        # print(x.shape)

        x = self.e8( x )
        x = self.bn8( x )
        x = self.lrelu( x )
        # print(x.shape)
        # print()

        # Decorder
        x = self.upsample2x( x )
        x = self.bn9_1( x )
        x = torch.cat( [ x7, x ], dim=1 )
        x = self.d1( x )
        x = self.bn9_2( x )
        x = self.dropout( x )
        x = self.relu( x )
        # print(x.shape)

        x = self.upsample2x( x )
        x = self.bn10_1( x )
        x = torch.cat( [ x6, x ], dim=1 )
        x = self.d2( x )
        x = self.bn10_2( x )
        x = self.dropout( x )
        x = self.relu( x )
        # print(x.shape)

        x = self.upsample2x( x )
        x = self.bn11_1( x )
        x = torch.cat( [ x5, x ], dim=1 )
        x = self.d3( x )
        x = self.bn11_2( x )
        x = self.dropout( x )
        x = self.relu( x )
        # print(x.shape)

        x = self.upsample2x( x )
        x = self.bn12_1( x )
        x = torch.cat( [ x4, x ], dim=1 )
        x = self.d4( x )
        x = self.bn12_2( x )
        x = self.relu( x )
        # print(x.shape)
        
        x = self.upsample32( x )
        x = self.bn13_1( x )
        x = torch.cat( [ x3, x ], dim=1 )
        x = self.d5( x )
        x = self.bn13_2( x )
        x = self.relu( x )
        # print(x.shape)

        x = self.upsample64( x )
        x = self.bn14_1( x )
        x = torch.cat( [ x2, x ], dim=1 )
        x = self.d6( x )
        x = self.bn14_2( x )
        x = self.relu( x )
        # print(x.shape)

        x = self.upsample128( x )
        x = self.bn15_1( x )
        x = torch.cat( [ x1, x ], dim=1 )
        x = self.d7( x )
        x = self.bn15_2( x )
        x = self.relu( x )
        # print(x.shape)

        x = self.upsample256( x )
        x = self.bn16_1( x )
        x = self.output( x )
        x = self.bn16_2( x ) 
        # print(x.shape)

        x = torch.squeeze( x )

        return x
    

def training( model, device, lr, train_X, train_Y, num_epochs=30, minibatch_size=64 ):
    # Data preproseccing
    train_dataset = torch.utils.data.TensorDataset( train_X, train_Y )

    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=minibatch_size, shuffle=True )

    loss_history = list()

    optimizer = optim.Adam( model.parameters(), lr=lr, betas=(0.5, 0.9) )
    lsd = LSD_loss()
    
    for epoch in range( num_epochs ):
        model.train()

        # data[0] : noisy, data[1] : clean
        for i, data in enumerate( train_loader ):
            x, y = data[ 0 ].to( device ), data[ 1 ].to( device )
            optimizer.zero_grad()
            output = model( x )
            loss = lsd( output, y )
            loss.backward()
            optimizer.step()
            loss_history.append( loss.item() )
        
            if not i % 50:
                print('Epoch : ', epoch + 1, '| batch size : ', i, ' / ', len(train_dataset) // minibatch_size ,' | train loss: %.4f' % loss.data.cpu().numpy())

    return model, loss_history