use crate::blob::{BlobType, Blob, ArcBlob};
use crate::layer::{LayerImpl, BlobVec};
use crate::proto::caffe::{TransformationParameter, LayerParameter};
use crate::data_transformer::DataTransformer;
use crate::util::blocking_queue::BlockingQueue;
use crate::internal_thread::{InternalThread, CancelToken, InternalThreadImpl};


/// Provides base for data layers that feed blobs to the Net.
pub struct BaseDataLayerImpl<T: BlobType> {
    pub layer: LayerImpl<T>,
    pub transform_param: TransformationParameter,
    pub data_transformer: Option<DataTransformer<T>>,
    pub output_labels: bool,
}

impl<T: BlobType> BaseDataLayerImpl<T> {
    pub fn new(param: &LayerParameter) -> Self {
        BaseDataLayerImpl {
            layer: LayerImpl::new(param),
            transform_param: param.get_transform_param().clone(),
            data_transformer: None,
            output_labels: false,
        }
    }
}

pub trait BaseDataLayer<T: BlobType> {
    fn get_data_impl(&self) -> &BaseDataLayerImpl<T>;

    fn get_data_impl_mut(&mut self) -> &mut BaseDataLayerImpl<T>;

    fn data_layer_setup(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>);
}

impl<T: BlobType> dyn BaseDataLayer<T> {
    /// LayerSetUp: implements common data layer setup functionality, and calls
    /// `data_layer_setUp` to do special data layer setup for individual layer types.
    /// This method may not be overridden except by the `BasePrefetchingDataLayer`.
    pub fn layer_setup(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        {
            let data = self.get_data_impl_mut();
            data.output_labels = top.len() != 1;
            data.data_transformer = Some(DataTransformer::new(&data.transform_param, data.layer.phase));
            data.data_transformer.as_mut().unwrap().init_rand();
        }
        // The subclasses should setup the size of bottom and top
        self.data_layer_setup(bottom, top);
    }
}


pub struct Batch<T: BlobType> {
    pub data: ArcBlob<T>,
    pub label: ArcBlob<T>,
}

pub struct BasePrefetchingDataLayerImpl<T: BlobType> {
    pub base: BaseDataLayerImpl<T>,

    pub prefetch: Vec<Batch<T>>,
    pub prefetch_free: BlockingQueue<Batch<T>>,
    pub prefetch_full: BlockingQueue<Batch<T>>,
    pub prefetch_current: Option<Batch<T>>,

    pub transformed_data: Blob<T>,
}

pub trait BasePrefetchingDataLayer<T: BlobType>: BaseDataLayer<T> {
    type SyncType: Sync + 'static;

    fn get_prefetch(&self) -> &BasePrefetchingDataLayerImpl<T>;

    fn get_prefetch_mut(&mut self) -> &mut BasePrefetchingDataLayerImpl<T>;

    fn get_sync_data(&self) -> &Self::SyncType;

    fn load_batch(data: &Self::SyncType, batch: &mut Batch<T>) where Self: Sized;
}

impl<T: BlobType, S: Sync + 'static> dyn BasePrefetchingDataLayer<T, SyncType = S> {
    pub fn layer_setup(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        //
    }

    pub fn forward_cpu(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        //
    }

    pub fn forward_gpu(&mut self, bottom: &BlobVec<T>, top: &BlobVec<T>) {
        //
    }
}

pub struct ThreadEntryData<T: BlobType> {
    pub prefetch_free: BlockingQueue<Batch<T>>,
    pub prefetch_full: BlockingQueue<Batch<T>>,
}

impl<T, U> InternalThread for dyn BasePrefetchingDataLayer<T, SyncType = U>
    where
        T: BlobType,
        U: Sync + 'static {
    type EntryData = ThreadEntryData<T>;

    fn get_thread(&self) -> &InternalThreadImpl {
        todo!()
    }

    fn get_thread_mut(&mut self) -> &mut InternalThreadImpl {
        todo!()
    }

    fn get_entry_data(&mut self) -> Self::EntryData {
        todo!()
    }

    fn internal_thread_entry(token: CancelToken, data: Self::EntryData) where Self: Sized {
        todo!()
    }
}


